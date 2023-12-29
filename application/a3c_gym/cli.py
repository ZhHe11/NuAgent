from typing import Any, Type
from argparse import Namespace

import threading
import gym.spaces as spaces
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

from uniagent.models.a2c import ActorCritic
from uniagent.envs.gym_control import create_gym_control
from uniagent.envs.atari import create_atari_env
from uniagent.trainers.parameter_server import get_parameter_server

from application.a3c_gym.async_agent import AsyncAgent
from application.a3c_gym import atari_net


def get_actor_critic_cls(args) -> Type[nn.Module]:
    if args.task_type == "gym_control":
        return ActorCritic
    elif args.task_type == "atari":
        if args.use_lstm:
            return atari_net.AtariLSTMAC
        else:
            return atari_net.AtariAC
    else:
        raise NotImplementedError


def make_env_wrapper(args):
    def make_env():
        if args.task_type == "gym_control":
            return create_gym_control(args.env_name)
        elif args.task_type == "atari":
            return create_atari_env(args.env_name, scale_obs=True)
        else:
            raise NotImplementedError

    return make_env


def run_worker(
    args: Namespace,
    rank: int,
    world_size: int,
    ps_name: str,
    model_class: Type[nn.Module],
    observation_space: spaces.Space,
    action_space: spaces.Space,
    counter: mp.Value,
    lock: threading.Lock,
    log_dir: str,
    async_agent_cls: Type[AsyncAgent] = None,
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
        print(f"starting evaluation task for rank={rank}")
        agent.test(counter, lock)
    else:
        print(f"starting training task for rank={rank}")
        agent.train(counter, lock)

    print(f"Worker {rank} finished task execution")

    rpc.shutdown()
