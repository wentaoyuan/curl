import re
from collections import namedtuple, defaultdict
from itertools import product
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def mkdir_if_not_exists(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True)
    return path


def filter_single_item(line: str) -> List[str]:
    return re.sub(r'(:|,|\"|\{|\})', '', line).split()


class DirectoryManager:
    def __init__(self, root_dir: Path = Path.home() / '.curl'):
        self.root_dir = root_dir

    @property
    def figure_root_dir(self):
        return mkdir_if_not_exists(self.root_dir / 'figures')

    def figure_dir(self, exp_name: str):
        return mkdir_if_not_exists(self.figure_root_dir / exp_name)

    @property
    def log_root_dir(self):
        return mkdir_if_not_exists(self.root_dir / 'logs')

    def log_dir(self, exp_name: str):
        return mkdir_if_not_exists(self.log_root_dir / exp_name)


class ModeInformation:
    def __init__(self, mode: str, query: str, key_list: List[str]):
        self.mode = mode
        self.query = query
        self.key_list = key_list

    @property
    def log_filename(self):
        return '{}.log'.format(self.mode)

    @property
    def item_cls(self):
        return namedtuple('{}item'.format(self.mode), self.key_list)


dm = DirectoryManager()


class Visualizer(ModeInformation):
    def __init__(self, exp_name: str, mode: str, query: str, key_list: List[str]):
        ModeInformation.__init__(self, mode, query, key_list)
        self.exp_name = exp_name

    def visualize(self):
        log_files = self.collect_log_files()
        data_dict = {f: self.read_log_file(f) for f in log_files}
        self.visualize_data(data_dict)

    @property
    def log_dir(self) -> Path:
        return dm.log_dir(self.exp_name)

    def collect_log_files(self):
        return self._collect_log_files(self.log_dir)

    def _collect_log_files(self, p):
        paths = []
        candidates = p.glob('*')
        for p in candidates:
            if p.is_dir():
                paths += self._collect_log_files(p)
            elif p.name == self.log_filename:
                paths.append(p)
        return paths

    def read_log_file(self, in_path: Path):
        with open(str(in_path), 'r') as file:
            lines = file.read().splitlines()
        wgl = list(map(filter_single_item, lines))
        items = []
        for words in wgl:
            items.append({key: float(value) for key, value in zip(words[::2], words[1::2])})
        items = list(filter(lambda x: all(k in x for k in self.key_list), items))
        items = list(filter(lambda x: x['step'] <= 100000, items))
        items = list(map(lambda x: self.item_cls(*[x[key] for key in self.key_list]), items))
        return items

    def visualize_data(self, data_dict: Dict[Path, List[Any]]):
        columns = ['model', 'index'] + self.key_list
        items = []
        index_dict = defaultdict(int)
        for key, value in data_dict.items():
            model = key.parent.parent.stem
            index = index_dict[model]
            for item in value:
                items.append((model, index, *item))
            index_dict[model] += 1

        data_frame = pd.DataFrame(items, columns=columns)

        xticks = list(range(2, 11, 2))
        xvalues, xnames = zip(*[(i * 10000, '{}k'.format(10 * i)) for i in xticks])

        sns.set(style="ticks", color_codes=True)
        sns.relplot(x='step', y=self.query, kind='line', hue='model', ci='sd', data=data_frame)
        plt.suptitle(self.exp_name)
        plt.ylabel(self.query.replace('_', ' '))
        plt.xlabel('steps')
        plt.xticks(list(xvalues), list(xnames))
        plt.savefig(str(dm.figure_root_dir / '{}_{}.png'.format(self.exp_name, self.mode)))
        plt.close()


def fetch_visualizer(exp_name: str, mode: str) -> Visualizer:
    if mode == 'train':
        query = 'episode_reward'
        key_list = ['episode_reward', 'episode', 'duration', 'step']
    elif mode == 'eval':
        query = 'mean_episode_reward'
        key_list = ['episode_reward', 'episode', 'mean_episode_reward', 'best_episode_reward', 'step']
    else:
        raise TypeError('invalid mode: {}'.format(mode))
    return Visualizer(exp_name, mode, query, key_list)


if __name__ == '__main__':
    exp_names = ['cartpole-swingup', 'walker-walk']
    modes = ['train', 'eval']

    for exp_name, mode in product(exp_names, modes):
        visualizer = fetch_visualizer(exp_name, mode)
        visualizer.visualize()
