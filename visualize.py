import re
from collections import namedtuple, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def mkdir_if_not_exists(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True)
    return path


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

modes = ['train', 'eval']
_queries = ['episode_reward', 'mean_episode_reward']
_key_lists = [['episode_reward', 'episode', 'duration', 'step'],
              ['episode_reward', 'episode', 'mean_episode_reward', 'best_episode_reward', 'step']]
_mode_info_dict = {m: ModeInformation(m, q, k) for m, q, k in zip(modes, _queries, _key_lists)}


def _collect_files(query_dir: Path, query_str: str) -> List[Path]:
    paths = []
    candidates = query_dir.glob('*')
    for p in candidates:
        if p.is_dir():
            paths += _collect_files(p, query_str)
        elif p.name == query_str:
            paths.append(p)
    return paths


def collect_files(query_dir: Path, mode: str) -> List[Path]:
    return _collect_files(query_dir, _mode_info_dict[mode].log_filename)


def filter_by(files: List[Path], query: str) -> List[Path]:
    return list(filter(lambda x: re.findall(query, str(x)), files))


def filter_single_item(line: str) -> List[str]:
    return re.sub(r'(:|,|\"|\{|\})', '', line).split()


def _read_log_file(in_path: Path, mode: str):
    key_list = _mode_info_dict[mode].key_list
    item_cls = _mode_info_dict[mode].item_cls
    with open(str(in_path), 'r') as file:
        lines = file.read().splitlines()
    wgl = list(map(filter_single_item, lines))
    items = []
    for words in wgl:
        items.append({key: float(value) for key, value in zip(words[::2], words[1::2])})
    items = list(filter(lambda x: all(k in x for k in key_list), items))
    items = list(filter(lambda x: x['step'] <= 100000, items))
    items = list(map(lambda x: item_cls(*[x[key] for key in key_list]), items))
    return items


def visualize_data(exp_name: str, data_dict: Dict[Path, List[Any]], mode: str):
    query = _mode_info_dict[mode].query
    key_list = _mode_info_dict[mode].key_list
    columns = ['model', 'index'] + key_list
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
    sns.relplot(x='step', y=query, kind='line', hue='model', ci='sd', data=data_frame)
    plt.ylabel(query.replace('_', ' '))
    plt.xlabel('steps')
    plt.xticks(list(xvalues), list(xnames))
    plt.savefig(str(dm.figure_root_dir / '{}_{}.png'.format(exp_name, mode)))
    plt.close()


if __name__ == '__main__':
    exp_names = ['cartpole-swingup', 'walker-walk']
    for exp_name in exp_names:
        log_dir = dm.log_dir(exp_name)
        for mode in modes:
            in_files = collect_files(log_dir, mode)
            data_dict = {f: _read_log_file(f, mode) for f in in_files}
            visualize_data(exp_name, data_dict, mode)
