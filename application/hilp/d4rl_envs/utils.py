import d4rl
import gym
import numpy as np

from dataset import Dataset


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


def get_dataset(
    env: gym.Env,
    env_name: str,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    dataset: Dataset = None,
    filter_terminals: bool = False,
    obs_dtype: np.dtype = np.float32,
    goal_conditioned: bool = True,
) -> Dataset:
    """Load D4RL dataset and return a new dataset which may be modified by given conditions.

    Args:
        env (gym.Env): Environment instance
        env_name (str): Environment name, used for environment filtering, cause' the trajectory end and done float will be determined by this
        clip_to_eps (bool, optional): Whether action clipping. Defaults to True.
        eps (float, optional): The tolerance of action clipping, being activated if `clip_to_eps` is True. Defaults to 1e-5.
        dataset (Any, optional): A dataset. Defaults to None.
        filter_terminals (bool, optional): Whether to drop terminal states. Defaults to False.
        obs_dtype (np.float32, optional): Data type for casting. Defaults to np.float32.
        goal_conditioned (bool, optional): Goal-conditioned mode or not. If true, the trajectory will be truncated, and add a row of masking to form a new dataset. Defaults to True.

    Returns:
        Dataset: _description_
    """
    if dataset is None:
        dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    if goal_conditioned:
        dataset["terminals"][-1] = 1

    if filter_terminals:
        # drop terminal transitions
        non_last_idx = np.nonzero(~dataset["terminals"])[0]
        last_idx = np.nonzero(dataset["terminals"])[0]
        penult_idx = last_idx - 1
        new_dataset = dict()
        for k, v in dataset.items():
            if k == "terminals":
                v[penult_idx] = 1
            new_dataset[k] = v[non_last_idx]
        dataset = new_dataset

    if "antmaze" in env_name:
        dones_float = np.zeros_like(dataset["rewards"])
        traj_ends = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            traj_end = (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
            )
            traj_ends[i] = traj_end
            if goal_conditioned:
                dones_float[i] = int(traj_end)
            else:
                dones_float[i] = int(traj_end or dataset["terminals"][i] == 1.0)
        dones_float[-1] = 1
        traj_ends[-1] = 1
    else:
        dones_float = dataset["terminals"].copy()
        traj_ends = dataset["terminals"].copy()

    observations = dataset["observations"].astype(obs_dtype)
    next_observations = dataset["next_observations"].astype(obs_dtype)

    if goal_conditioned:
        masks = 1.0 - dones_float
    else:
        masks = 1.0 - dataset["terminals"].astype(np.float32)

    return Dataset.create(
        observations=observations,
        actions=dataset["actions"].astype(np.float32),
        rewards=dataset["rewards"].astype(np.float32),
        masks=masks,
        dones_float=dones_float.astype(np.float32),
        next_observations=next_observations,
        traj_ends=traj_ends,
    )


def get_normalization(dataset):
    returns = []
    ret = 0
    for r, term in zip(dataset["rewards"], dataset["dones_float"]):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


def normalize_dataset(env_name, dataset):
    if "antmaze" in env_name:
        return dataset.copy({"rewards": dataset["rewards"] - 1.0})
    else:
        normalizing_factor = get_normalization(dataset)
        dataset = dataset.copy({"rewards": dataset["rewards"] / normalizing_factor})
        return dataset


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
