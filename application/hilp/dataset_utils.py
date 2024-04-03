from typing import Tuple, Any, Dict
import os

# mute D4RL warnings
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import numpy as np
import jax

from jax import numpy as jnp

from d4rl_envs import utils as d4rl_utils
from d4rl_envs import ant as d4rl_ant
from d4rl_envs import ant_diagnostics

from uniagent.envs.utils import EpisodeMonitor
from dataset import Dataset


@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        v1, v2 = agent.network(
            jax.tree_map(lambda x: x[None], s),
            jax.tree_map(lambda x: x[None], g),
            method="value",
        )
        return (v1 + v2) / 2

    observations = trajectory["observations"]
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(
        observations, observations
    )
    return {
        "dist_to_beginning": all_values[:, 0],
        "dist_to_end": all_values[:, -1],
        "dist_to_middle": all_values[:, all_values.shape[1] // 2],
    }


@jax.jit
def get_v_goal(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    v1, v2 = agent.network(observations, goal, method="value")
    return (v1 + v2) / 2


def get_env_and_dataset(
    env_name: str, width: int, height: int, discount: float
) -> Tuple[Any, Dataset, Any, Dict]:
    """Return a tuple of (environment_instance, dataset, auxilary_environment, preset_goals)

    Returns:
        Tuple[Env, Dataset, Env, Dict]: A tuple, the dict is a description of the goal.
    """

    aux_env = {}
    goal_info = {}

    if "antmaze" in env_name:
        env_name = env_name

        if "ultra" in env_name:
            import d4rl_ext
            import gym

            env = gym.make(env_name)
            env = EpisodeMonitor(env)
        else:
            env = d4rl_utils.make_env(env_name)

        dataset = d4rl_utils.get_dataset(env, env_name, goal_conditioned=True)

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

            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
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
            dataset = d4rl_utils.get_dataset(
                env, env_name, dataset=dataset, filter_terminals=True
            )
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
            dataset = d4rl_utils.get_dataset(env, env_name, filter_terminals=True)
            dataset = dataset.copy(
                {
                    "observations": dataset["observations"][:, :30],
                    "next_observations": dataset["next_observations"][:, :30],
                }
            )
    else:
        raise NotImplementedError

    return env, dataset, aux_env, goal_info
