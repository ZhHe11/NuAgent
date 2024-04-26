import os
import datetime
import tempfile

from argparse import Namespace

import wandb
import tqdm
import torch

from torch.utils.tensorboard import SummaryWriter

from uniagent.data.collector import Collector
from uniagent.utils.wandb import setup_wandb

from .cmd_utils import get_exp_name, set_seed
from .envs import make_env
from .envs.evaluation import eval_random_option_generation


def main(args: Namespace):
    global_start_time = int(datetime.datetime.now().timestamp())
    exp_name, _ = get_exp_name(args, global_start_time)

    if "WANDB_API_KEY" in os.environ:
        wandb_output_dir = tempfile.mkdtemp()
        wandb.init(
            project="metra",
            entity="",
            group=args.run_group,
            name=exp_name,
            config=vars(args),
            dir=wandb_output_dir,
        )

    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)

    set_seed(args.seed)

    env = make_env(args, args.max_path_length)

    print("==== Task Information ====")
    print("* Task name: ", args.env_name)
    print("* Observation space: ", env.observation_space)
    print("* Action space: ", env.action_space)

    obs_dim = env.spec.observation_space.flat_dim
    action_dim = env.spec.action_space.flat_dim

    args.wandb["project"] = "hilp_gcrl"
    args.wandb["name"] = args.wandb["exp_descriptor"] = exp_name
    args.wandb["group"] = args.wandb["exp_prefix"] = args.run_group

    if args.use_wandb:
        setup_wandb(args, dict(), **args.wandb)

        args.save_dir = os.path.join(
            args.save_dir,
            wandb.run.project,
            wandb.config.exp_prefix,
            wandb.config.experiment_id,
        )

    writer = SummaryWriter(args.save_dir)
    if args.algo == "metra":
        from .learner import MetraAgent, create_replay_buffer

        agent = MetraAgent(
            args,
            obs_dim=obs_dim,
            goal_dim=args.option_dim,
            act_dim=action_dim,
            action_space=env.action_space,
            load_path=args.restore_path,
        )
        buffer = create_replay_buffer(args, env)
    elif args.algo == "dads":
        from .baselines.dads import DADSAgent, create_replay_buffer

        agent = DADSAgent(
            args, obs_dim=obs_dim, goal_dim=args.option_dim, act_dim=action_dim
        )
        buffer = create_replay_buffer(args, env)

    agent.to(args.device)

    def action_interface_wrapper(agent):
        def f(observation):
            option = agent.sample_option(observation)
            action = agent.sample_action(observation, option)
            return option, action

        return f

    collector = Collector(buffer, env, action_interface_wrapper(agent))

    info = {"ave_return": 0}
    tprocess = tqdm.tqdm(
        range(1, args.n_epochs + 1),
        smoothing=0.1,
        dynamic_ncols=True,
        desc="Training METRA",
        leave=True,
    )
    for i in tprocess:
        agent.train()
        collector.collect(args.batch_size, args.seed, args.max_path_length)
        batch = collector.sample(args.batch_size, to_torch=True, device=args.device)
        loss_info = agent.run(batch, action_space=env.spec.action_space)
        loss_info["buffer_size"] = len(buffer)

        if i % args.log_interval == 0:
            if args.use_wandb:
                wandb.log(loss_info, step=i)
            else:
                pass

        if i == 1 or i % args.eval_interval == 0:
            agent.eval()
            eval_metrics = eval_random_option_generation(args, env, agent)
            info.update(eval_metrics)
            if args.use_wandb:
                wandb.log(eval_metrics, step=i)
            for k, v in info.items():
                writer.add_scalar(k, v, i)

        if i % args.save_interval == 0:
            pass

        str_info = " ".join([f"{k}: {v:.3f}" for k, v in info.items()])
        tprocess.set_postfix_str(str_info)
        for k, v in loss_info.items():
            writer.add_scalar(k, v, i)
