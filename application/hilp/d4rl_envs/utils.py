import d4rl
import gym
import numpy as np


def make_env(env_name: str):
    from uniagent.envs.utils import EpisodeMonitor

    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env


def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None,]

    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.0

    if n_cols is None:
        if v.shape[0] <= 4:
            n_cols = 2
        elif v.shape[0] <= 9:
            n_cols = 3
        elif v.shape[0] <= 16:
            n_cols = 4
        else:
            n_cols = 6
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def save_video(label, step, tensor, fps=15, n_cols=None):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t

    if tensor.dtype in [object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    # tensor: (t, h, w, c)
    tensor = tensor.transpose(0, 3, 1, 2)
    return wandb.Video(tensor, fps=15, format="mp4")


def record_video(label, step, renders=None, n_cols=None, skip_frames=1):
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        renders[i] = np.concatenate(
            [
                render,
                np.zeros(
                    (max_length - render.shape[0], *render.shape[1:]),
                    dtype=render.dtype,
                ),
            ],
            axis=0,
        )
        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders)
    return save_video(label, step, renders, n_cols=n_cols)


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine

    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.86, lookat=[-0.3, 0.5, 2.0], azimuth=90, elevation=-60)
    img = camera.render()
    return img


import wandb


class CsvLogger:
    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row["step"] = step
        if self.file is None:
            self.file = open(self.path, "w")
            if self.header is None:
                self.header = [
                    k
                    for k, v in row.items()
                    if not isinstance(v, self.disallowed_types)
                ]
                self.file.write(",".join(self.header) + "\n")
            filtered_row = {
                k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)
            }
            self.file.write(
                ",".join([str(filtered_row.get(k, "")) for k in self.header]) + "\n"
            )
        else:
            filtered_row = {
                k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)
            }
            self.file.write(
                ",".join([str(filtered_row.get(k, "")) for k in self.header]) + "\n"
            )
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
