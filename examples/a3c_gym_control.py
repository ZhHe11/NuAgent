import os
import argparse
import shutup

shutup.please()

from itertools import count

import torch
import torch.multiprocessing as mp

from uniagent.trainers.optimizers import SharedAdam
from uniagent.models.a2c import ActorCritic
from uniagent.envs.gym_control import create_gym_control

from application.a3c_gym.train import train
from application.a3c_gym.eval import test


parser = argparse.ArgumentParser(description="A3C for Atari")
parser.add_argument(
    "--lr",
    type=float,
    default=0.0003,  # try LogUniform(1e-4.5, 1e-3.5)
    help="learning rate",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.95,
    help="worker discount factor for rewards",
)
parser.add_argument(
    "--llambda", type=float, default=0.95, help="parameter for GAE (worker only)"
)
parser.add_argument(
    "--entropy-coef",
    type=float,
    default=0.01,
    help="entropy term coefficient (also called beta)",
)
parser.add_argument(
    "--value-loss-coef",
    type=float,
    default=1,
    help="worker value loss coefficient",
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
    help="number of forward steps in A3C (every `num_steps`, do a backward step)",
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


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # if not torch.cuda.is_available() else "0"
    mp.set_start_method("spawn")

    args = parser.parse_args()

    args.task_type = "control"
    args.device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.use_cuda
        else torch.device("cpu")
    )

    torch.manual_seed(args.seed)
    env = create_gym_control(args.env_name)
    print(
        f"env: {args.env_name}\nobservation_space: {env.observation_space}\naction_space: {env.action_space}"
    )

    shared_model = ActorCritic(env.observation_space, env.action_space)
    shared_model.to(args.device)

    shared_model.share_memory()

    if args.no_shared:
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
        train(rank, shared_model, counter, log_dir, lock, optimizer, args)
    else:
        p = mp.Process(
            target=test,
            args=(args.num_processes, shared_model, counter, log_dir, lock, args),
        )
        p.start()
        processes.append(p)

        for rank in range(0, args.num_processes):
            p = mp.Process(
                target=train,
                args=(rank, shared_model, counter, log_dir, lock, optimizer, args),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
