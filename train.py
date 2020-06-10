import argparse
import json
import os
import time

import numpy as np
import torch

import dmc2gym
import utils
from argument import Argument, MultiViewEncoderType
from curl_sac import CurlSacAgent
from logger import Logger
from video import VideoRecorder

logger = utils.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    # parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    # parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--camera_ids', nargs='+', default=[0], type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    # parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    # parser.add_argument('--save_video', default=False, action='store_true')
    # parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--load_buffer', default=False, action='store_true')
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--restore_train_step', default=0, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--multi_view_encoder_str', default='pool', type=str)
    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, args: Argument):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args: Argument, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            args=args
        )
    else:
        assert 'agent is not supported: {}'.format(args.agent)


def main():
    user_args = parse_args()
    args = Argument(user_args)
    utils.set_seed_everywhere(args.seed)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat,
        camera_ids=args.camera_ids,
        multi_view_encoder_type=args.multi_view_encoder_type,
    )
    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        stacked = len(args.camera_ids) > 1 and args.multi_view_encoder_type == MultiViewEncoderType.Pool
        env = utils.FrameStack(env, k=args.frame_stack, stacked=stacked)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    env_name = args.domain_name + '-' + args.task_name
    exp_strs = [env_name, ts,
                'im{}'.format(args.image_size),
                'b{}'.format(args.batch_size),
                's{}'.format(args.seed),
                'c{}'.format('_'.join([str(v) for v in args.camera_ids])),
                args.encoder_type]
    if len(args.camera_ids) > 1:
        exp_strs.append(str(args.multi_view_encoder_type))
    exp_name = '-'.join(exp_strs)
    logger.info(exp_name)

    args.work_dir = args.work_dir + '/' + exp_name

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None, camera_ids=args.camera_ids)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        pre_aug_obs_shape = utils.fetch_obs_shape(args, pre_aug=True)
        obs_shape = utils.fetch_obs_shape(args, pre_aug=False)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )
    if args.load_buffer:
        replay_buffer.load(buffer_dir)

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )
    if args.load_model:
        if args.encoder_type == 'pixel':
            agent.load_curl(model_dir, args.restore_train_step)
        agent.load(model_dir, args.restore_train_step)

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    if args.restore_train_step == 0:
        L.log('eval/episode', episode, 0)
        evaluate(env, agent, video, args.num_eval_episodes, L, 0, args)
    for step in range(args.restore_train_step, args.num_train_steps):
        # evaluate agent periodically
        if (step + 1) % args.eval_freq == 0:
            L.log('eval/episode', episode, step + 1)
            evaluate(env, agent, video, args.num_eval_episodes, L, step + 1, args)
            if args.save_model:
                if args.encoder_type == 'pixel':
                    agent.save_curl(model_dir, step + 1)
                agent.save(model_dir, step + 1)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
