from typing import List, Tuple, Any

from itertools import count
from argparse import Namespace
from collections import namedtuple

import time
import copy
import random
import numpy as np
import torch
import torch.optim as optim

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from uniagent.envs.atari import create_atari_env

from .policy import Agent
from .eval import rollout

from tensorboardX import SummaryWriter


Batch = namedtuple(
    "Batch", "obs, action, reward, next_obs, done, hx, cx, next_hx, next_cx"
)


def ensure_shared_grads(model: nn.Module, shared_model: nn.Module):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is None:
            shared_param._grad = param.grad


def stack_list_to_tensor(data: List[Any], device="cpu") -> torch.Tensor:
    if isinstance(data[0], torch.Tensor):
        return torch.stack([e.detach() for e in data]).squeeze(1).to(device)
    elif isinstance(data[0], np.ndarray):
        return torch.from_numpy(np.stack(data)).to(device)
    else:
        return torch.from_numpy(np.asarray(data)).to(device)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


def train(
    rank: int,
    env: Any,
    shared_model: nn.Module,
    counter: int,
    log_dir: str,
    lock,
    optimizer: optim.Optimizer,
    args: Namespace,
    model_cls: nn.Module,
):
    seed = args.seed + rank
    torch.manual_seed(seed)

    # env = create_atari_env(args.env_name, args.max_episode_length)
    if hasattr(env, "seed"):
        env.seed(seed)

    agent = Agent(model_cls, env.observation_space, env.action_space, args.device)
    if not args.async_mode:
        del agent.model
        agent.model = shared_model
        agent.target_model = copy.deepcopy(shared_model)

    if optimizer is None:
        print("no shared optimizer, use local optimizer")
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir)
    agent.model.train()

    buffer = ReplayBuffer(args.replay_buffer_size)
    start_time = time.time()

    for epoch in count():
        if args.async_mode:
            agent.model.load_state_dict(shared_model.state_dict())

        net_states = agent.init_state(1, args.device)
        done = False
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float().to(args.device)

        rewards = []

        for step in count():
            action, new_net_states = agent.act(obs.unsqueeze(0), net_states)

            next_obs, reward, done, truncated, info = env.step(action.item())

            done = done or truncated

            with lock:
                counter.value += 1

            rewards.append(reward)
            buffer.push(
                obs.cpu().numpy(),
                action.item(),
                reward,
                next_obs,
                done,
                *net_states,
                *new_net_states
            )

            net_states = new_net_states

            if len(buffer) >= args.batch_size:
                list_of_tuple = buffer.sample(args.batch_size)
                batch = Batch(*map(stack_list_to_tensor, zip(*list_of_tuple)))

                next_qs, _ = agent.compute_target_q(
                    batch.next_obs.float(),
                    (batch.next_hx.float(), batch.next_cx.float()),
                )
                q, _ = agent.compute_q(
                    batch.obs.float(), (batch.hx.float(), batch.cx.float())
                )

                assert next_qs.shape == q.shape

                if args.double_q:
                    with torch.no_grad():
                        next_qs_with_eval, _ = agent.compute_q(
                            batch.next_obs.float(),
                            (batch.next_hx.float(), batch.next_cx.float()),
                        )

                    next_action = next_qs_with_eval.argmax(-1, keepdim=True)
                else:
                    next_action = next_qs.argmax(-1, keepdim=True)

                assert isinstance(next_qs, torch.Tensor)
                target_q = batch.reward.float() + args.gamma * next_qs.gather(
                    -1, next_action
                ).squeeze(-1) * (1.0 - batch.done.float())

                q_action = q.gather(-1, batch.action.unsqueeze(-1).long()).squeeze(-1)

                assert q_action.shape == target_q.shape, (
                    q_action.shape,
                    target_q.shape,
                )

                td_error = F.mse_loss(
                    q_action,
                    target_q,
                )

                optimizer.zero_grad()
                td_error.backward()
                if args.async_mode:
                    ensure_shared_grads(agent.model, shared_model)
                optimizer.step()
                agent.soft_update()

                with lock:
                    writer.add_scalar(
                        "train/td_error" + str(rank), td_error.item(), counter.value
                    )
                    writer.add_scalar("train/eps" + str(rank), agent.eps, counter.value)
                    writer.add_scalar(
                        "train/reward_mean", batch.reward.mean().item(), counter.value
                    )
                    writer.add_scalars(
                        "train/qs" + str(rank),
                        {"q": q.mean().item(), "target_q": target_q.mean().item()},
                        counter.value,
                    )

            if done:
                with lock:
                    writer.add_scalar(
                        "train/episode_length" + str(rank), step, counter.value
                    )
                    writer.add_scalar(
                        "train/episode_reward" + str(rank),
                        np.sum(rewards),
                        counter.value,
                    )
                break

            obs = torch.from_numpy(next_obs).to(args.device)
        if not args.async_mode:
            rollout(epoch, counter, agent, env, start_time, writer, args)
