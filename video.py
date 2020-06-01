import imageio
import os
import numpy as np


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_ids=(0, ), fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_ids = camera_ids
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            views = []
            for camera_id in self.camera_ids:
                try:
                    frame = env.render(
                        mode='rgb_array',
                        height=self.height,
                        width=self.width,
                        camera_id=camera_id
                    )
                except:
                    frame = env.render(
                        mode='rgb_array',
                    )
                views.append(frame)
            frame = np.concatenate(views, axis=1)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
