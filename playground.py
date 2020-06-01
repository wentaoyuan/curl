# # from dm_control import suite
# # import numpy as np
# #
# # # Load one task:
# # env = suite.load(domain_name="cartpole", task_name="swingup")
# #
# # # Iterate over a task set:
# # for domain_name, task_name in suite.BENCHMARKING:
# #     env = suite.load(domain_name, task_name)
# #
# # # Step through an episode and print out reward, discount and observation.
# # action_spec = env.action_spec()
# # time_step = env.reset()
# # while not time_step.last():
# #     action = np.random.uniform(action_spec.minimum,
# #                                action_spec.maximum,
# #                                size=action_spec.shape)
# #     time_step = env.step(action)
# #     print(time_step.reward, time_step.discount, time_step.observation)
# #
#
#
#
# # from dm_control import suite
# # from dm_control import viewer
# # import numpy as np
# #
# # env = suite.load(domain_name="humanoid", task_name="stand")
# # action_spec = env.action_spec()
# #
# # # Define a uniform random policy.
# # def random_policy(time_step):
# #   del time_step  # Unused.
# #   return np.random.uniform(low=action_spec.minimum,
# #                            high=action_spec.maximum,
# #                            size=action_spec.shape)
# #
# # # Launch the viewer application.
# # viewer.launch(env, policy=random_policy)
#
#
# from dm_control import suite
# from dm_control.suite.wrappers import pixels
#
# env = suite.load('hopper', 'hop')
#
# wrapped_env = pixels.Wrapper(env, render_kwargs={'camera_id': 'cam0'})

import cv2
from dm_control import suite
from dm_control.suite.wrappers import pixels

import numpy as np

# Load one task:
env = suite.load(domain_name='walker', task_name='walk')
# wrapped_env = pixels.Wrapper(env, render_kwargs={'camera_id': 0})

# Iterate over a task set:
# for domain_name, task_name in suite.BENCHMARKING:
#     env = suite.load(domain_name, task_name)

# Step through an episode and print out reward, discount and observation.

height, width = 480, 480

action_spec = env.action_spec()
time_step = env.reset()
image = env.physics.render(height, width, camera_id=0)
images = []
for i in range(100):
# while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
    time_step = env.step(action)
    image1 = env.physics.render(height, width, camera_id=0)
    image2 = env.physics.render(height, width, camera_id=1)
    image = np.zeros((image1.shape[0], image1.shape[1] * 2, image1.shape[2]), dtype=image1.dtype)
    image[:, :image1.shape[1], :] = image1.copy()
    image[:, image1.shape[1]:, :] = image2.copy()
    images.append(image)
    # image3 = env.physics.render(height, width, camera_id=2)

    # time_step = wrapped_env.step(action)
    # cv2.imshow('image', image[..., ::-1])
    # cv2.imshow('image2', image2)
    # cv2.imshow('image3', image3)
    # cv2.waitKey()
    # print(time_step.reward, time_step.discount, time_step.observation['pixels'].shape)


import imageio
from pathlib import Path
video_path = Path.home() / '.curl/walker-walk.mp4'
imageio.mimsave(str(video_path), images, fps=24)



# # Copyright 2017 The dm_control Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #    http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ============================================================================
#
# """Demonstration of amc parsing for CMU mocap database.
#
# To run the demo, supply a path to a `.amc` file:
#
#     python mocap_demo --filename='path/to/mocap.amc'
#
# CMU motion capture clips are available at mocap.cs.cmu.edu
# """
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import time
# # Internal dependencies.
#
# from absl import app
# from absl import flags
#
# from dm_control.suite import humanoid_CMU
# from dm_control.suite.utils import parse_amc
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# FLAGS = flags.FLAGS
# flags.DEFINE_string('filename', None, 'amc file to be converted.')
# flags.DEFINE_integer('max_num_frames', 90,
#                      'Maximum number of frames for plotting/playback')
#
#
# def main(unused_argv):
#   env = humanoid_CMU.stand()
#
#   # Parse and convert specified clip.
#   converted = parse_amc.convert(FLAGS.filename,
#                                 env.physics, env.control_timestep())
#
#   max_frame = min(FLAGS.max_num_frames, converted.qpos.shape[1] - 1)
#
#   width = 480
#   height = 480
#   video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)
#
#   for i in range(max_frame):
#     p_i = converted.qpos[:, i]
#     with env.physics.reset_context():
#       env.physics.data.qpos[:] = p_i
#     video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
#                           env.physics.render(height, width, camera_id=1)])
#
#   tic = time.time()
#   for i in range(max_frame):
#     if i == 0:
#       img = plt.imshow(video[i])
#     else:
#       img.set_data(video[i])
#     toc = time.time()
#     clock_dt = toc - tic
#     tic = time.time()
#     # Real-time playback not always possible as clock_dt > .03
#     plt.pause(max(0.01, 0.03 - clock_dt))  # Need min display time > 0.0.
#     plt.draw()
#   plt.waitforbuttonpress()
#
#
# if __name__ == '__main__':
#   flags.mark_flag_as_required('filename')
#   app.run(main)