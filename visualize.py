import re
from collections import namedtuple, defaultdict
from itertools import product, chain
from pathlib import Path
from typing import List, Dict, Any, Tuple

from pandas import DataFrame
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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


Scalar = namedtuple('Scalar', ['timestamp', 'step', 'value'])
ExpInfo = namedtuple('ExpInfo', ['env_name', 'task_name', 'seed', 'camera_indices', 'model_type'])


def fetch_exp_info_from_name(name: str):
    words = name.split('-')
    env_name, task_name = words[0], words[1]
    seed = int(words[6][1:])
    camera_indices = [int(v) for v in re.split(r'_|,', words[7][1:])]
    model_type = '-'.join(words[8:])
    return ExpInfo(env_name, task_name, seed, camera_indices, model_type)


def tensorboard_visualizer(root_dir: Path) -> List[Tuple[ExpInfo, List[Scalar]]]:
    files = root_dir.glob('events.out*')

    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'histograms': 1
    }
    keyword = 'eval/episode_reward'

    values = []
    for file in files:
        exp_info = fetch_exp_info_from_name(file.parent.parent.stem)
        event_acc = EventAccumulator(str(file), tf_size_guidance)
        event_acc.Reload()
        if not keyword in event_acc.Tags()['scalars']:
            return []
        rewards = event_acc.Scalars(keyword)
        rewards = list(map(lambda x: Scalar(*x), rewards))
        values.append((exp_info, rewards))
    return values


if __name__ == '__main__':
    path_list = [
        Path.home() / '.curl/walker-walk-06-10-im84-b128-s0-c0_1-pixel-stack/tb',
        Path.home() / '.curl/walker-walk-06-10-im84-b128-s0-c0_1-pixel-pool/tb',
        Path.home() / '.curl/logs/walker-walk/dual-stack/walker-walk-06-01-im84-b128-s100-n0,1-pixel-stack0/tb',
        Path.home() / '.curl/logs/walker-walk/dual-pool/walker-walk-06-01-im84-b128-s1100-n0,1-pixel-pool0/tb',
        Path.home() / '.curl/logs/walker-walk/dual-pool/walker-walk-06-01-im84-b128-s1400-n0,1-pixel-pool0/tb',
        Path.home() / '.curl/walker-walk-06-10-im84-b128-s10000-n0,1-pixel-pool1/tb',
        Path.home() / '.curl/walker-walk-06-10-im84-b128-s10001-n0,1-pixel-pool1/tb',
    ]
    value_list = list(chain.from_iterable([tensorboard_visualizer(p) for p in path_list]))

    raw_data_dict = defaultdict(list)
    for exp_info, values in value_list:
        key = '{}-{}'.format(exp_info.env_name, exp_info.task_name)
        unique_key = 'c{}-{}'.format(len(exp_info.camera_indices), exp_info.model_type)
        data = [(exp_info.seed, unique_key, v.step, v.value) for v in values]
        raw_data_dict[key] += data

    dfs = []
    max_step = 0
    for key, data_list in raw_data_dict.items():
        seeds, types, steps, values = zip(*data_list)
        data_dict = dict()
        data_dict['key'] = [key] * len(seeds)
        data_dict['seed'] = list(seeds)
        data_dict['type'] = list(types)
        data_dict['step'] = list(steps)
        data_dict['reward'] = list(values)
        df = DataFrame.from_dict(data=data_dict)
        dfs.append(df)

    df = pd.concat(dfs)

    font_family = 'Lato'
    font_weight = 'medium'
    # rcparams: http://omz-software.com/pythonista/matplotlib/users/customizing.html
    rc_params = {
        'font.family': font_family,
        'font.weight': font_weight,
        'axes.labelweight': font_weight,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'axes.titlepad': 10,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
    }
    palette = sns.color_palette('Blues')
    sns.set_palette(palette)
    sns.set(style='whitegrid', rc=rc_params)

    # hue_order = sorted(unique_key_set)
    # tick_unit = 5000
    # max_tick = int(round(max_step / tick_unit))
    # xticks = list(range(1, max_tick + 1))
    # xvalues, xnames = zip(*[(i * tick_unit, '{}k'.format(i * (tick_unit // 1000))) for i in xticks])
    f = sns.relplot(x='step', y='reward', kind='line', ci='sd', hue='type', col='key', data=df)
                    # palette=sns.color_palette('colorblind', n_colors=len(unique_key_set)))
    # for axis, num_agent_str in zip(f.fig.axes, ['two', 'three', 'four']):
    #     axis.set_title('{} agents'.format(num_agent_str), fontdict={'fontweight': font_weight, 'fontsize': 17})
    # plt.ylim(top=1, bottom=-3)
    # plt.xticks(list(xvalues), list(xnames))
    f.savefig(str(Path.home() / '.curl/reward.png'))

    # exp_names = ['cartpole-swingup', 'walker-walk']
    # modes = ['train', 'eval']
    #
    # for exp_name, mode in product(exp_names, modes):
    #     visualizer = fetch_visualizer(exp_name, mode)
    #     visualizer.visualize()
