from typing import Dict, Any
from collections import defaultdict

import gym
import numpy as np

from tqdm import trange

from .d4rl_envs.utils import kitchen_render


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def env_reset(env_name, env, goal_info, base_observation, policy_type):
    observation, done = env.reset(), False
    if policy_type == "random_skill" and "antmaze" in env_name:
        observation[:2] = [20, 8]
        env.set_state(observation[:15], observation[15:])

    if "antmaze" in env_name:
        goal = env.wrapped_env.target_goal
        obs_goal = np.concatenate([goal, base_observation[-27:]])
    elif "kitchen" in env_name:
        if "visual" in env_name:
            observation = kitchen_render(env)
            obs_goal = goal_info["ob"]
        else:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
    else:
        raise NotImplementedError

    return observation, obs_goal


def env_step(env_name, env, action):
    if "antmaze" in env_name:
        next_observation, reward, done, info = env.step(action)
    elif "kitchen" in env_name:
        next_observation, reward, done, info = env.step(action)
        if "visual" in env_name:
            next_observation = kitchen_render(env)
        else:
            next_observation = next_observation[:30]
    else:
        raise NotImplementedError

    return next_observation, reward, done, info


def get_frame(env_name, env):
    if "antmaze" in env_name:
        size = 200
        cur_frame = (
            env.render(mode="rgb_array", width=size, height=size)
            .transpose(2, 0, 1)
            .copy()
        )
    elif "kitchen" in env_name:
        cur_frame = kitchen_render(env, wh=100).transpose(2, 0, 1)
    else:
        raise NotImplementedError
    return cur_frame


def add_episode_info(env_name, env, info, trajectory):
    if "antmaze" in env_name:
        info["final_dist"] = np.linalg.norm(
            trajectory["next_observation"][-1][:2] - env.wrapped_env.target_goal
        )
    elif "kitchen" in env_name:
        info["success"] = float(info["episode"]["return"] == 4.0)
    else:
        raise NotImplementedError


from .learner import HILPAgent


def evaluate_with_trajectories(
    agent: HILPAgent,
    env: gym.Env,
    goal_info: Dict[str, Any],
    env_name: str,
    num_episodes: int,
    base_observation: np.ndarray = None,
    num_video_episodes: int = 0,
    policy_type: str = "goal_skill",
    planning_info: Dict[str, Any] = None,
) -> Dict[str, float]:
    policy_fn = agent.sample_skill_actions

    if policy_type == "goal_skill_planning":
        planning_info["examples"]["phis"] = np.array(
            agent.get_phi(planning_info["examples"]["observations"])
        )

    trajectories = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_episodes + num_video_episodes):
        trajectory = defaultdict(list)

        observation, obs_goal = env_reset(
            env_name, env, goal_info, base_observation, policy_type
        )
        done = False

        render = []
        step = 0
        skill = None

        while not done:
            policy_obs = observation
            policy_goal = obs_goal

            if policy_type == "goal_skill":
                phi_obs, phi_goal = agent.get_phi(np.array([policy_obs, policy_goal]))
                skill = (phi_goal - phi_obs) / np.linalg.norm(phi_goal - phi_obs)
                action = policy_fn(
                    observations=policy_obs, skills=skill, temperature=0.0
                )
            elif policy_type == "goal_skill_planning":
                phi_obs, phi_goal = agent.get_phi(np.array([policy_obs, policy_goal]))

                for k in range(planning_info["num_recursions"]):
                    ex_phis = planning_info["examples"]["phis"]
                    dists_s = np.linalg.norm(ex_phis - phi_obs, axis=-1)
                    dists_g = np.linalg.norm(ex_phis - phi_goal, axis=-1)
                    dists_diff = np.maximum(dists_s, dists_g)
                    way_idxs = dists_diff.argsort()
                    phi_goal = ex_phis[way_idxs[: planning_info["num_knns"]]].mean(
                        axis=0
                    )
                way_skill = (phi_goal - phi_obs) / np.linalg.norm(phi_goal - phi_obs)
                action = policy_fn(
                    observations=policy_obs, skills=way_skill, temperature=0.0
                )
            else:
                raise NotImplementedError

            action = np.array(action)
            next_observation, reward, done, info = env_step(env_name, env, action)

            step += 1

            # Render
            if i >= num_episodes and step % 3 == 0:
                cur_frame = get_frame(env_name, env)
                render.append(cur_frame)
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                skill=skill,
                info=info,
            )
            if i < num_episodes:
                add_to(trajectory, transition)
                add_to(stats, flatten(info))
            observation = next_observation
        if i < num_episodes:
            add_episode_info(env_name, env, info, trajectory)
            add_to(stats, flatten(info, parent_key="final"))
            trajectories.append(trajectory)
        else:
            renders.append(np.array(render))

    scalar_stats = {}
    for k, v in stats.items():
        scalar_stats[k] = np.mean(v)
    return scalar_stats, trajectories, renders
