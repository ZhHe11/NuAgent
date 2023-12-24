import os
import argparse
import threading
import shutup

shutup.please()

from itertools import count

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

from uniagent.models.a2c import ActorCritic
from uniagent.envs.gym_control import create_gym_control
from uniagent.trainers.parameter_server import (
    get_parameter_server,
    run_parameter_server,
)

from application.a3c_gym.async_agent import AsyncAgent


def make_env_wrapper(args):
    def make_env():
        return create_gym_control(args.env_name)

    return make_env


def run_worker(
    args,
    rank,
    world_size,
    ps_name,
    model_class,
    observation_space,
    action_space,
    counter,
    lock,
    log_dir,
):
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(name=f"trainer_{rank}", rank=rank, world_size=world_size)

    print(f"Worker {rank} done initializing RPC")

    model_kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
    }

    param_server_rref = rpc.remote(
        ps_name, get_parameter_server, args=(model_class, model_kwargs, world_size - 2)
    )

    print("* fetched parameter server reference", param_server_rref)

    agent = AsyncAgent(
        param_server_rref,
        rank,
        model_class,
        model_kwargs,
        make_env_wrapper(args),
        device=args.device,
    )

    if rank == 1:
        print("starting evaluation task")
        agent.test(args, counter, lock, log_dir)
    else:
        print("starting training task")
        agent.train(args, counter, lock, log_dir)

    print(f"Worker {rank} finished task execution")

    rpc.shutdown()


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
parser.add_argument("--master-addr", default="localhost")
parser.add_argument("--master-port", default="29500")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # if not torch.cuda.is_available() else "0"
    mp.set_start_method("spawn")

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

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

    processes = []

    counter = mp.Value("i", 0)
    lock = mp.Lock()

    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join("runs/test", current_time + "_" + socket.gethostname())
    ps_name = "parameter_server"

    p = mp.Process(target=run_parameter_server, args=(0, args.num_processes, ps_name))
    p.start()
    processes.append(p)

    for rank in range(1, args.num_processes):
        p = mp.Process(
            target=run_worker,
            args=(
                args,
                rank,
                args.num_processes,
                ps_name,
                ActorCritic,
                env.observation_space,
                env.action_space,
                counter,
                lock,
                log_dir,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
