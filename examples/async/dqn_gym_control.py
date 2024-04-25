import os
import argparse
import shutup

shutup.please()

from itertools import count

import gym
import torch
import torch.multiprocessing as mp

from gym.wrappers.transform_reward import TransformReward
from uniagent.trainers.optimizers import SharedAdam

from application.dqn_gym.train import train
from application.dqn_gym.eval import test
from application.dqn_gym.policy import DQN, MoeDQN


parser = argparse.ArgumentParser(description="DQN for Gym control")
parser.add_argument(
    "--lr",
    type=float,
    default=2.5e-4,  # try LogUniform(1e-4.5, 1e-3.5)
    help="learning rate",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="worker discount factor for rewards",
)
parser.add_argument(
    "--max-grad-norm", type=float, default=50, help="value loss coefficient"
)
parser.add_argument("--seed", type=int, default=123, help="random seed")
parser.add_argument(
    "--num-processes", type=int, default=4, help="how many training processes to use"
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=400,
    help="number of forward steps in DQN (every `num_steps`, do a backward step)",
)
parser.add_argument(
    "--max-episode-length",
    type=int,
    default=1000000,
    help="maximum length of an episode",
)
parser.add_argument(
    "--env-name",
    default="CartPole-v1",
    help="environment to train on (default: CartPole-v1)",
)
parser.add_argument(
    "--no-shared", action="store_true", help="use an optimizer without shared momentum."
)
parser.add_argument(
    "--async-mode", action="store_true", help="use async mode for evaluation"
)
parser.add_argument("--channel-first", default=True, help="use channel first input")
parser.add_argument("--use-cuda", action="store_true")
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--replay-buffer-size", default=100000, type=int)
parser.add_argument("--double-q", action="store_true")
parser.add_argument("--backbone", type=str, default="mlp", choices={"mlp", "moe"})


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # if not torch.cuda.is_available() else "0"
    mp.set_start_method("spawn")

    args = parser.parse_args()

    args.device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.use_cuda
        else torch.device("cpu")
    )

    torch.manual_seed(args.seed)
    env = TransformReward(gym.make(args.env_name), lambda x: 0.1 * x)
    print(
        f"env: {args.env_name}\nobservation_space: {env.observation_space}\naction_space: {env.action_space}"
    )

    model_cls = DQN if args.backbone == "mlp" else MoeDQN

    shared_model = model_cls(env.observation_space, env.action_space)
    shared_model.to(args.device)

    if args.async_mode:
        shared_model.share_memory()

    if args.no_shared or not args.async_mode:
        optimizer = None
    else:
        optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value("i", 0)
    lock = mp.Lock()

    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join("runs", current_time + "_" + socket.gethostname())

    if not args.async_mode:
        rank = 0
        train(
            rank, env, shared_model, counter, log_dir, lock, optimizer, args, model_cls
        )
    else:
        p = mp.Process(
            target=test,
            args=(
                args.num_processes,
                env,
                shared_model,
                counter,
                log_dir,
                lock,
                args,
                model_cls,
            ),
        )
        p.start()
        processes.append(p)

        for rank in range(0, args.num_processes):
            p = mp.Process(
                target=train,
                args=(
                    rank,
                    env,
                    shared_model,
                    counter,
                    log_dir,
                    lock,
                    optimizer,
                    args,
                    model_cls,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
