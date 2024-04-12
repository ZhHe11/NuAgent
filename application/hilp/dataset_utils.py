from typing import Tuple, Any, Dict
import os

# mute D4RL warnings
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import numpy as np
import d4rl
import gym
import tree
import torch

from uniagent.envs.utils import EpisodeMonitor

from .d4rl_envs import utils as d4rl_utils
from .d4rl_envs import ant as d4rl_ant
from .d4rl_envs import ant_diagnostics

from .dataset import Dataset


# @jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        v = agent(
            "value",
            tree.map_structure(lambda x: x[None], s),
            tree.map_structure(lambda x: x[None], g),
        ).mean(0)
        return v

    observations = tree.map_structure(
        lambda x: torch.from_numpy(x).float().to(agent.config.device),
        trajectory["observations"],
    )
    with torch.no_grad():
        all_values = torch.vmap(
            torch.vmap(get_v, in_dims=(None, 0)), in_dims=(0, None)
        )(observations, observations)
        all_values = all_values.cpu().numpy()
    return {
        "dist_to_beginning": all_values[:, 0],
        "dist_to_end": all_values[:, -1],
        "dist_to_middle": all_values[:, all_values.shape[1] // 2],
    }


# @jax.jit
def get_v_goal(agent, goal, observations):
    goal = np.tile(goal, (observations.shape[0], 1))
    # v1, v2 = agent.network(observations, goal, method="value")
    observations = tree.map_structure(
        lambda x: torch.from_numpy(x).float().to(agent.config.device), observations
    )
    goal = tree.map_structure(
        lambda x: torch.from_numpy(x).float().to(agent.config.device), goal
    )
    with torch.no_grad():
        v = agent("value", observations, goal).mean(0)
    return v.cpu().numpy()


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


from argparse import Namespace


def get_env_and_dataset(args: Namespace) -> Tuple[Any, Dataset, Any, Dict]:
    """Return a tuple of (environment_instance, dataset, auxilary_environment, preset_goals)

    Returns:
        Tuple[Env, Dataset, Env, Dict]: A tuple, the dict is a description of the goal.
    """

    aux_env = {}
    goal_info = {}

    env_name = args.env_name
    width = args.width
    height = args.height
    discount = args.discount

    if "antmaze" in env_name:
        env_name = env_name

        if "ultra" in env_name:
            import d4rl_ext
            import gym

            env = gym.make(env_name)
            env = EpisodeMonitor(env)
        else:
            env = d4rl_utils.make_env(env_name)

        dataset = get_dataset(env, env_name, goal_conditioned=True)

        # be sure the reward range from 0 to 1
        assert (
            dataset["rewards"].min() == 0 and dataset["rewards"].max() == 1.0
        ), "be sure the reward ranges from 0 to 1, given min={}, max={}".format(
            dataset["rewards"].min(), dataset["rewards"].max()
        )

        # convert reward to goal reaching mode
        dataset = dataset.copy({"rewards": dataset["rewards"] - 1.0})
        env.render(mode="rgb_array", width=width, height=height)
        if "large" in env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90

            viz_env, viz_data = d4rl_ant.get_env_and_dataset(env_name)
            # convert to hilp dataset
            viz_dataset = Dataset.create(**viz_data)
            viz = ant_diagnostics.Visualizer(
                env_name, viz_env, viz_dataset, discount=discount
            )
            init_state = np.copy(viz_dataset["observations"][0])
            init_state[:2] = (12.5, 8)
            aux_env = {
                "viz_env": viz_env,
                "viz_dataset": viz_dataset,
                "viz": viz,
            }
        elif "ultra" in env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
            env.viewer.cam.elevation = -90
        else:
            raise NotImplementedError
    elif "kitchen" in env_name:
        if "visual" in env_name:
            orig_env_name = env_name.split("visual-")[1]
            env = d4rl_utils.make_env(orig_env_name)
            dataset = dict(np.load(f"data/d4rl_kitchen_rendered/{orig_env_name}.npz"))
            dataset = get_dataset(env, env_name, dataset=dataset, filter_terminals=True)
            if "partial" in env_name:
                # Precomputed index for a goal state
                goal_info = {
                    "ob": dataset["observations"][118319],
                }
            elif "mixed" in env_name:
                from d4rl_envs.utils import kitchen_render

                state = env.reset()
                # This is dataset['observations'][118319] of kitchen-partial-v0
                goal_state = [
                    -2.3403780e00,
                    -1.3053924e00,
                    1.1021180e00,
                    -1.8613019e00,
                    1.5087037e-01,
                    1.7687809e00,
                    1.2525779e00,
                    2.9698312e-02,
                    3.0899283e-02,
                    3.9908718e-04,
                    4.9550228e-05,
                    -1.9946630e-05,
                    2.7519276e-05,
                    4.8786267e-05,
                    3.2835731e-05,
                    2.6504624e-05,
                    3.8422750e-05,
                    -6.9888681e-01,
                    -5.0150707e-02,
                    3.4855098e-01,
                    -9.8701166e-03,
                    -7.6958216e-03,
                    -8.0031347e-01,
                    -1.9142720e-01,
                    7.2064394e-01,
                    1.6191028e00,
                    1.0021452e00,
                    -3.2998802e-04,
                    3.7205056e-05,
                    5.3616576e-02,
                ]
                # Set the goal state for kitchen-mixed-v0
                goal_state[9:] = state[39:]
                env.sim.set_state(np.concatenate([goal_state, env.init_qvel]))
                env.sim.forward()
                goal_info = {
                    "ob": kitchen_render(env).astype(np.float32),
                }
                env.reset()
        else:
            env = d4rl_utils.make_env(env_name)
            dataset = get_dataset(env, env_name, filter_terminals=True)
            dataset = dataset.copy(
                {
                    "observations": dataset["observations"][:, :30],
                    "next_observations": dataset["next_observations"][:, :30],
                }
            )
    else:
        raise NotImplementedError

    return env, dataset, aux_env, goal_info
