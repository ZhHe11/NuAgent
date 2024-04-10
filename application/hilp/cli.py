import os
import time
import glob
import pickle
import datetime

from functools import partial
from argparse import ArgumentParser, Namespace

import numpy as np
import wandb
import tqdm
import tree

from torch import nn
from uniagent.utils.wandb import setup_wandb, default_wandb_config

from .d4rl_envs.utils import record_video, CsvLogger
from .d4rl_envs import ant as d4rl_ant
from .dataset_utils import get_env_and_dataset, get_traj_v, get_v_goal
from .dataset import GCDataset
from . import utils
from .learner import HILPAgent
from .evaluation import evaluate_with_trajectories


def main(args: Namespace):
    g_start_time = int(datetime.datetime.now().timestamp())

    exp_name = ""
    exp_name += f"sd{args.seed:03d}_"
    if "SLURM_JOB_ID" in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if "SLURM_PROCID" in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if "SLURM_RESTART_COUNT" in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f"{g_start_time}"
    exp_name += f'_{args.wandb["name"]}'
    exp_name += f"_rnd_{args.use_rnd}"

    args.wandb["project"] = "hilp_gcrl"
    args.wandb["name"] = args.wandb["exp_descriptor"] = exp_name
    args.wandb["group"] = args.wandb["exp_prefix"] = args.run_group
    setup_wandb(args, dict(), **args.wandb)

    args.save_dir = os.path.join(
        args.save_dir,
        wandb.run.project,
        wandb.config.exp_prefix,
        wandb.config.experiment_id,
    )
    os.makedirs(args.save_dir, exist_ok=True)

    # constructing goal-conditioned dataset
    env, dataset, aux_env, goal_info = get_env_and_dataset(
        args.env_name, args.width, args.height, args.discount
    )

    # considering some observations maybe a tree of numpy ndarray
    base_observation = tree.map_structure(lambda x: x[0], dataset["observations"])
    env.reset()

    train_dataset = GCDataset(
        dataset=dataset,
        p_currgoal=args.p_currgoal,
        p_trajgoal=args.p_trajgoal,
        p_randomgoal=args.p_randomgoal,
        discount=args.discount,
        p_aug=args.p_aug,
    )

    sample = train_dataset.sample(1)

    agent: nn.Module = HILPAgent(
        args,
        sample["observations"].shape[-1],
        sample["goals"].shape[-1],
        sample["actions"].shape[-1],
    )
    agent.to(args.device)

    if args.restore_path is not None:
        restore_path = args.restore_path
        candidates = glob.glob(restore_path)
        if len(candidates) == 0:
            raise Exception(f"Path does not exist: {restore_path}")
        if len(candidates) > 1:
            raise Exception(f"Multiple matching paths exist for: {restore_path}")
        if args.restore_epoch is None:
            restore_path = candidates[0] + "/params.pkl"
        else:
            restore_path = candidates[0] + f"/params_{args.restore_epoch}.pkl"
        # TODO(ming): load training agent from local storage
        agent = HILPAgent.load_from(restore_path)
        print(f"Restored from {restore_path}")

    if "antmaze" in args.env_name:
        example_trajectory = train_dataset.sample(
            50, indx=np.arange(1000, 1050), evaluation=True
        )
    else:
        example_trajectory = train_dataset.sample(
            50, indx=np.arange(0, 50), evaluation=True
        )

    train_logger = CsvLogger(os.path.join(args.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(args.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()

    for i in tqdm.tqdm(
        range(1, args.total_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        batch = train_dataset.sample(args.batch_size)

        if args.use_rnd:
            raise NotImplementedError
        else:
            update_info = agent.update(batch)

        if i % args.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / args.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i == 1 or i % args.eval_interval == 0:
            eval_metrics = {}
            trajs_dict = {}
            for policy_type in (
                ["goal_skill", "goal_skill_planning"]
                if args.planning_num_recursions > 0
                else ["goal_skill"]
            ):
                num_episodes = args.eval_episodes
                num_video_episodes = args.num_video_episodes

                if policy_type == "goal_skill_planning":
                    planning_info = dict(
                        num_recursions=args.planning_num_recursions,
                        num_knns=args.planning_num_knns,
                        examples=dataset.sample(args.planning_num_states),
                    )
                else:
                    planning_info = None
                eval_info, cur_trajs, renders = evaluate_with_trajectories(
                    agent,
                    env,
                    goal_info=goal_info,
                    env_name=args.env_name,
                    num_episodes=num_episodes,
                    base_observation=base_observation,
                    num_video_episodes=num_video_episodes,
                    policy_type=policy_type,
                    planning_info=planning_info,
                )
                eval_metrics.update(
                    {f"{policy_type}/{k}": v for k, v in eval_info.items()}
                )
                trajs_dict[policy_type] = cur_trajs

            if ArgumentParser.num_video_episodes > 0:
                video = record_video("Video", i, renders=renders)
                eval_metrics["video"] = video

            traj_metrics = get_traj_v(agent, example_trajectory)
            value_viz = utils.make_visual_no_image(
                traj_metrics,
                [
                    partial(utils.visualize_metric, metric_name=k)
                    for k in traj_metrics.keys()
                ],
            )
            eval_metrics["value_traj_viz"] = wandb.Image(value_viz)

            if "antmaze" in args.env_name and "large" in args.env_name:
                trajs = trajs_dict["goal_skill"]
                viz_env, viz_dataset, viz = (
                    aux_env["viz_env"],
                    aux_env["viz_dataset"],
                    aux_env["viz"],
                )
                traj_image = d4rl_ant.trajectory_image(viz_env, viz_dataset, trajs)
                eval_metrics["trajectories"] = wandb.Image(traj_image)

                new_metrics_dist = viz.get_distance_metrics(trajs)
                eval_metrics.update(
                    {f"debugging/{k}": v for k, v in new_metrics_dist.items()}
                )

                image_goal = d4rl_ant.gcvalue_image(
                    viz_env,
                    viz_dataset,
                    partial(get_v_goal, agent),
                )
                eval_metrics["v_goal"] = wandb.Image(image_goal)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        if i % args.save_interval == 0:
            pass
            # save_dict = dict(
            #     agent=flax.serialization.to_state_dict(agent),
            # )

            # fname = os.path.join(args.save_dir, f"params_{i}.pkl")
            # print(f"Saving to {fname}")
            # with open(fname, "wb") as f:
            #     pickle.dump(save_dict, f)
    train_logger.close()
    eval_logger.close()


def get_command_parser():
    parser = ArgumentParser("Training HILP")

    parser.add_argument("--env-name", type=str, help="environment name", required=True)
    parser.add_argument("--width", type=int, default=200, help="window size, the width")
    parser.add_argument(
        "--height", type=int, default=200, help="window size, the height"
    )

    parser.add_argument(
        "--save-dir", type=str, default="exp/", help="experiment logging directory"
    )
    parser.add_argument("--restore-path", type=str, default=None)
    parser.add_argument(
        "--run-group", type=str, default="debug", help="naming experiment group"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--num-video-episodes", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--save-interval", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--total-steps", type=int, default=1000000)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.95)
    parser.add_argument("--use-layer-norm", type=int, default=1)
    parser.add_argument("--skill-dim", type=int, default=32)
    parser.add_argument("--skill-expectile", type=float, default=0.9)
    parser.add_argument("--skill-temperature", type=float, default=10.0)
    parser.add_argument("--skill-discount", type=float, default=0.99)
    parser.add_argument("--p-currgoal", type=float, default=0.0)
    parser.add_argument("--p-trajgoal", type=float, default=0.625)
    parser.add_argument("--p-randomgoal", type=float, default=0.375)

    parser.add_argument("--planning-num-recursions", type=int, default=0)
    parser.add_argument("--planning-num_states", type=int, default=50000)
    parser.add_argument("--planning-num-knns", type=int, default=50)

    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--p-aug", type=float, default=None)
    parser.add_argument("--use-rnd", type=int, default=0)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    args.wandb = default_wandb_config()

    return args
