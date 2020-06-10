import gym
import numpy as np
from dm_control import suite
from dm_env import specs
from gym import core, spaces

from argument import MultiViewEncoderType

gym.logger.set_level(40)


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(
            self,
            domain_name,
            task_name,
            task_kwargs=None,
            visualize_reward={},
            from_pixels=False,
            height=84,
            width=84,
            camera_ids=(0, ),
            frame_skip=1,
            environment_kwargs=None,
            channels_first=True,
            multi_view_encoder_type=MultiViewEncoderType.Stack
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_ids = camera_ids
        self._frame_skip = frame_skip
        self._channels_first = channels_first
        self._multi_view_encoder_type = multi_view_encoder_type

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        self._state_space = _spec_to_box(
            self._env.observation_spec().values()
        )

        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    @property
    def num_camera(self):
        return len(self._camera_ids)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = []
            for camera_id in self._camera_ids:
                obs.append(self.render(
                    height=self._height,
                    width=self._width,
                    camera_id=camera_id
                ))
            if self._multi_view_encoder_type == MultiViewEncoderType.Pool:
                obs = np.stack(obs, axis=0)  # H, W, C > n, H, W, C
                if self._channels_first:
                    obs = obs.transpose(0, 3, 1, 2)  # C, H, W > n, C, H, W
            elif self._multi_view_encoder_type == MultiViewEncoderType.Stack:
                obs = np.concatenate(obs, axis=-1)
                if self._channels_first:
                    obs = obs.transpose(2, 0, 1)
            else:
                raise TypeError('invalid multi_view_encoder_type: {}'.format(self._multi_view_encoder_type))
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed=None):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_ids[0]
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
