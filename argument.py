from enum import Enum
import numpy as np


class PrintableEnum(Enum):
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)


class MultiViewEncoderType(PrintableEnum):
    Pool = 0
    Stack = 1


def fetch_multi_view_encoder_type(type_str: str):
    if type_str == 'pool':
        return MultiViewEncoderType.Pool
    elif type_str == 'stack':
        return MultiViewEncoderType.Stack
    else:
        raise TypeError('invalid multi view encoder type: {}'.format(type_str))


class Argument:
    def __init__(self, user_args):
        self.domain_name = 'cheetah'
        self.task_name = 'run'
        self.pre_transform_image_size = 100
        self.image_size = 84
        self.action_repeat = 1
        self.frame_stack = 3
        self.replay_buffer_capacity = 100000
        self.camera_ids = [0]

        self.init_steps = 1000
        self.num_train_steps = 100000
        self.batch_size = 128
        self.eval_freq = 10000
        self.num_eval_episodes = 10

        self.agent = 'curl_sac'
        self.hidden_dim = 1024
        self.critic_lr = 1e-3
        self.critic_beta = 0.9
        self.critic_tau = 0.01
        self.critic_target_update_freq = 2
        self.actor_lr = 1e-3
        self.actor_beta = 0.9
        self.actor_log_std_min = -10
        self.actor_log_std_max = 2
        self.actor_update_freq = 2
        self.encoder_type = 'pixel'
        self.encoder_feature_dim = 50
        self.encoder_lr = 1e-3
        self.encoder_tau = 0.05
        self.num_layers = 4
        self.num_filters = 32
        self.curl_latent_dim = 128
        self.discount = 0.99
        self.init_temperature = 0.1
        self.alpha_lr = 1e-4
        self.alpha_beta = 0.5
        self.cpc_update_freq = 1

        self.seed = 1
        self.work_dir = '.'
        self.save_tb = True
        self.save_buffer = False
        self.save_video = True
        self.save_model = True
        self.detach_encoder = False
        self.load_buffer = False
        self.load_model = False
        self.restore_train_step = 0
        self.log_interval = 100
        self.multi_view_encoder_str = 'pool'

        # override from user args
        for key, value in user_args.__dict__.items():
            self.__dict__[key] = value

        if self.seed == -1:
            self.seed = np.random.randint(1, 1000000)

        if len(self.camera_ids) == 1:
            self.multi_view_encoder_str = 'stack'

    @property
    def multi_view_encoder_type(self):
        return fetch_multi_view_encoder_type(self.multi_view_encoder_str)
