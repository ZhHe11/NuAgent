import os
import datetime
import tempfile

from argparse import Namespace

import wandb
import tqdm
import torch


from uniagent.data.collector import Collector

from .cmd_utils import get_exp_name, set_seed
from .envs import make_env, eval_episodes


def main(args: Namespace):
    global_start_time = int(datetime.datetime.now().timestamp())
    exp_name, _ = get_exp_name(global_start_time)

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

    obs_dim = env.spec.observation_space.flat_dim
    action_dim = env.spec.action_space.flat_dim

    args.save_dir = os.path.join(
        args.save_dir,
        wandb.run.project,
        wandb.config.exp_prefix,
        wandb.config.experiment_id,
    )

    if args.algo == "metra":
        from .learner import MetraAgent, create_replay_buffer

        agent = MetraAgent(
            args,
            obs_dim=obs_dim,
            goal_dim=args.option_dim,
            act_dim=action_dim,
            load_path=args.load_path,
        )
        buffer = create_replay_buffer(env)
    elif args.algo == "dads":
        from .baselines.dads import DADSAgent, create_replay_buffer

        agent = DADSAgent(
            args, obs_dim=obs_dim, goal_dim=args.option_dim, act_dim=action_dim
        )
        buffer = create_replay_buffer(env)

    agent.to(args.device)

    def action_interface_wrapper(agent):
        def f(observation):
            with torch.no_grad():
                action = agent(observation)
            return action.cpu().numpy()

    collector = Collector(buffer, env, action_interface_wrapper(agent))

    for i in tqdm.tqdm(
        range(1, args.total_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        collector.collect(env)
        batch = collector.sample(args.batch_size, to_torch=True, device=args.device)
        loss_info = agent.update(batch, action_space=env.spec.action_space)

        if i % args.log_interval == 0:
            wandb.log(loss_info, step=i)

        if i == 1 or i % args.eval_interval == 0:
            eval_metrics = eval_episodes(env, action_interface_wrapper(agent))
            wandb.log(eval_metrics, step=i)

        if i % args.save_interval == 0:
            pass
