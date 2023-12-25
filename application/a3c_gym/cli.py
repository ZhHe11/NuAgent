from typing import Any

import torch.distributed.rpc as rpc

from uniagent.envs.gym_control import create_gym_control
from uniagent.envs.atari import create_atari_env
from uniagent.trainers.parameter_server import get_parameter_server

from application.a3c_gym.async_agent import AsyncAgent


def make_env_wrapper(args):
    def make_env():
        if args.task_type == "gym_control":
            return create_gym_control(args.env_name)
        elif args.task_type == "atari":
            return create_atari_env(args.env_name)
        else:
            raise NotImplementedError

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
    async_agent_cls: Any = None,
):
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(name=f"trainer_{rank}", rank=rank, world_size=world_size)

    print(f"Worker {rank} done initializing RPC")

    model_kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
    }

    # note the worker num is world_size - 2
    param_server_rref = rpc.remote(
        ps_name,
        get_parameter_server,
        args=(args, model_class, model_kwargs, world_size - 2, "avg"),
    )

    print("* fetched parameter server reference", param_server_rref)

    if async_agent_cls is None:
        async_agent_cls = AsyncAgent

    agent = async_agent_cls(
        args,
        param_server_rref,
        rank,
        model_class,
        model_kwargs,
        make_env_wrapper(args),
        log_dir,
    )

    if rank == 1:
        print("starting evaluation task")
        agent.test(counter, lock)
    else:
        print("starting training task")
        agent.train(counter, lock)

    print(f"Worker {rank} finished task execution")

    rpc.shutdown()
