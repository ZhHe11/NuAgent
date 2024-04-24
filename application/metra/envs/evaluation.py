from typing import Dict, Tuple, Callable
from argparse import Namespace

import gym
import torch
import numpy as np

from torch import nn
from sklearn import decomposition
from .plot import FigManager, draw_2d_gaussians, record_video


def get_2d_colors(points, min_point, max_point) -> np.ndarray:
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack(
        (
            colors,
            (2 - np.sum(colors, axis=1, keepdims=True)) / 2,
        )
    )
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors


def get_option_colors(options, color_range=4) -> np.ndarray:
    num_options = options.shape[0]
    dim_option = options.shape[1]

    if dim_option <= 2:
        # Use a predefined option color scheme
        if dim_option == 1:
            options_2d = []
            d = 2.0
            for i in range(len(options)):
                option = options[i][0]
                if option < 0:
                    abs_value = -option
                    options_2d.append((d - abs_value * d, d))
                else:
                    abs_value = option
                    options_2d.append((d, d - abs_value * d))
            options = np.array(options_2d)
        option_colors = get_2d_colors(
            options, (-color_range, -color_range), (color_range, color_range)
        )
    else:
        if dim_option > 3 and num_options >= 3:
            pca = decomposition.PCA(n_components=3)
            # Add random noises to break symmetry.
            pca_options = np.vstack((options, np.random.randn(dim_option, dim_option)))
            pca.fit(pca_options)
            option_colors = np.array(pca.transform(options))
        elif dim_option > 3 and num_options < 3:
            option_colors = options[:, :3]
        elif dim_option == 3:
            option_colors = options

        max_colors = np.array([color_range] * 3)
        min_colors = np.array([-color_range] * 3)
        if all((max_colors - min_colors) > 0):
            option_colors = (option_colors - min_colors) / (max_colors - min_colors)
        option_colors = np.clip(option_colors, 0, 1)

        option_colors = np.c_[option_colors, np.full(len(option_colors), 0.8)]

    return option_colors


def generate_options(args: Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.discrete_option:
        # generate options
        eye_options = np.eye(args.option_dim)
        random_options = []
        colors = []
        for i in range(args.option_dim):
            num_traj_per_option = args.num_eval_trajectories // args.option_dim + (
                i < args.num_eval_trajectories % args.option_dim
            )
            random_options_i = np.tile(
                eye_options[i], (num_traj_per_option, 1)
            ).tolist()
            random_options.extend(random_options_i)
            colors.extend([i] * num_traj_per_option)
        random_options = np.array(random_options)
        colors = np.array(colors)
        num_evals = len(random_options)

        from matplotlib import cm

        cmap = "tab10" if args.option_dim <= 10 else "tab20"
        random_option_colors = []

        random_option_colors.extend(
            [cm.get_cmap(cmap)(colors[i])[:3] for i in range(num_evals)]
        )
        random_option_colors = np.array(random_option_colors)
    else:
        random_options = np.random.randn(args.num_eval_trajectories, args.option_dim)
        if args.unit_length:
            random_options = random_options / np.linalg.norm(
                random_options, axis=1, keepdims=True
            )
        random_option_colors = get_option_colors(random_options * 4)
    return random_options, random_option_colors


from typing import Sequence, Any
from collections import namedtuple

import tree


Trajectory = namedtuple(
    "Trajectory", "observations,actions,rewards,dones,options,env_info"
)


def process_trajectories(agent: nn.Module, trajectories: Sequence[Trajectory]):
    ret = 0
    n = len(trajectories)
    for traj in trajectories:
        ret += sum(traj.rewards) / n
    statistic = {"ave_return": ret}
    env_infos = [traj.env_info for traj in trajectories]
    keys = list(env_infos[0][0].keys())
    rets = []
    for env_info in env_infos:
        _rets = {}
        for k in keys:
            _rets[k] = np.stack([x[k] for x in env_info])
        rets.append({"env_infos": _rets})
    return rets, statistic


def get_trajectories(
    args: Namespace, env: gym.Env, agent: nn.Module, options: Sequence[Any] = None
) -> Sequence[Trajectory]:
    trajectories = []
    num_eval_traj = len(options) if options is not None else args.num_eval_trajectories

    # distinguish from original options, for save
    for i in range(num_eval_traj):
        observations, actions, rewards, dones, option_traj, env_info = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        obs = env.reset()
        done = False

        while not done:
            observations.append(obs)
            if options is not None:
                option = options[i]
            else:
                option = agent.sample_option(obs)
            action = agent.sample_action(obs, option)
            next_obs, rew, done, info = env.step(action)
            rewards.append(rew)
            dones.append(done)
            option_traj.append(option)
            env_info.append(info)
            obs = next_obs

        # save last observation
        observations.append(obs)
        trajectories.append(
            Trajectory(observations, actions, rewards, dones, option_traj, env_info)
        )

    return trajectories


def eval_with_option_planner(args: Namespace, env: gym.Env, agent: nn.Module):
    trajectories = get_trajectories(args, env, agent)
    raise NotImplementedError


def eval_random_option_generation(
    args: Namespace, env: gym.Env, agent: nn.Module
) -> Dict[str, float]:
    random_options, option_colors = generate_options(args)

    trajectories = get_trajectories(
        args,
        env,
        agent,
        options=random_options,
    )

    # TODO(ming): parse trajectories here
    trajectories, info = process_trajectories(agent, trajectories)

    with FigManager(agent, "TrajPlot_RandomZ") as fm:
        assert hasattr(
            env, "render_trajectories"
        ), "please ensure you have implemented `render_trajectories` for your environment for rendering"
        env.render_trajectories(
            trajectories, option_colors, agent.config.eval_plot_axis, fm.ax
        )

    return info

    # last_obs = torch.stack([torch.from_numpy(v.observations[-1]).to(agent.config.device) for v in trajectories])
    # option_dists = agent.traj_encoder(last_obs)

    # option_means = option_dists.mean.detach().cpu().numpy()
    # if agent.config.use_inner_product:
    #     option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
    # else:
    #     option_stddevs = option_dists.stddev.detach().cpu().numpy()
    # option_samples = option_dists.mean.detach().cpu().numpy()

    # with FigManager(agent, f'PhiPlot') as fm:
    #     draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
    #     draw_2d_gaussians(
    #         option_samples,
    #         [[0.03, 0.03]] * len(option_samples),
    #         option_colors,
    #         fm.ax,
    #         fill=True,
    #         use_adaptive_axis=True,
    #     )

    # eval_option_metrics = {}

    # # for video saving
    # if agent.config.eval_record_video:
    #     if agent.config.discrete_goal:
    #         video_options = np.eye(agent.config.option_dim)
    #         video_options = video_options.repeat(agent.config.num_video_repeats, axis=0)
    #     else:
    #         if agent.config.option_dim == 2:
    #             radius = 1. if agent.config.unit_length else 1.5
    #             video_options = []
    #             for angle in [3, 2, 1, 4]:
    #                 video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
    #             video_options.append([0, 0])
    #             for angle in [0, 5, 6, 7]:
    #                 video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
    #             video_options = np.array(video_options)
    #         else:
    #             video_options = np.random.randn(9, agent.config.option_dim)
    #             if agent.config.unit_length:
    #                 video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
    #         video_options = video_options.repeat(agent.config.num_video_repeats, axis=0)
    #     video_trajectories = get_trajectories(
    #         agent,
    #         options=generate_options(video_options),
    #         action_interface=action_interface
    #     )
    #     record_video(agent, 'Video_RandomZ', video_trajectories, skip_frames=agent.config.video_skip_frames)

    # eval_option_metrics.update(env.calc_eval_metrics(trajectories, is_option_trajectories=True))
