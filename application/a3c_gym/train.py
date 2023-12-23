from itertools import count
from argparse import Namespace

import time
import random
import numpy as np
import torch
import torch.optim as optim

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from uniagent.envs.atari import create_atari_env
from uniagent.envs.gym_control import create_gym_control
from application.a3c_gym.agent import Agent

from .eval import rollout

from tensorboardX import SummaryWriter


def ensure_shared_grads(model: nn.Module, shared_model: nn.Module):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is None:
            shared_param._grad = param.grad


def train(
    rank: int,
    shared_model: nn.Module,
    counter: int,
    log_dir: str,
    lock,
    optimizer: optim.Optimizer,
    args: Namespace,
):
    seed = args.seed + rank
    torch.manual_seed(seed)

    if args.task_type == "atari":
        env = create_atari_env(
            args.env_name, args.max_episode_length, use_reward_clip=False
        )
    elif args.task_type == "control":
        env = create_gym_control(args.env_name)

    if hasattr(env, "seed"):
        env.seed(args.seed + rank)

    agent = Agent(
        shared_model.__class__, env.observation_space, env.action_space, args.device
    )
    if not args.async_mode:
        agent.model = shared_model

    if optimizer is None:
        print("no shared optimizer")
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=log_dir)

    agent.model.train()

    episode_length = 0
    start_time = time.time()
    done = True

    for epoch in count():
        time.sleep(random.random())
        # Sync with the shared model
        if args.async_mode:
            agent.model.load_state_dict(shared_model.state_dict())

        if done:
            net_state = agent.init_state(1, args.device)
        else:
            net_state = tuple(map(lambda x: x.detach(), net_state))

        values = []
        log_probs = []
        rewards = []
        entropies = []  # regularisation

        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(args.device)

        for step in range(args.num_steps):
            episode_length += 1
            value, action, log_prob, entropy, net_state = agent.act(
                obs.unsqueeze(0), net_state
            )
            entropies.append(entropy.squeeze())
            values.append(value.squeeze())
            log_probs.append(log_prob.squeeze())

            next_obs, reward, done, truncated, info = env.step(action.cpu().numpy()[0])

            done = done or truncated

            with lock:
                counter.value += 1

            rewards.append(reward)

            if done:
                break

            obs = torch.from_numpy(next_obs).to(args.device)

        if not done:
            value, _, _, _, _ = agent.act(obs.unsqueeze(0), net_state)
            value = value.detach().squeeze()
        else:
            value = torch.zeros(1).to(args.device).squeeze()

        with lock:
            writer.add_scalars(
                "training/episode_info" + str(rank),
                {
                    "episode_reward": sum(rewards),
                    "episode_length": len(rewards),
                },
                epoch,
            )

        values.append(value)

        gae = 0.0
        traj_len = len(rewards)
        ret = values[-1].item()
        R = [0.0] * traj_len
        GAE = [0.0] * traj_len

        for i in reversed(range(traj_len)):
            ret = args.gamma * ret + rewards[i]
            delta_t = (
                rewards[i]
                + args.gamma * values[i + 1].cpu().item()
                - values[i].cpu().item()
            )
            gae = gae * args.gamma * args.llambda + delta_t

            GAE[i] = gae
            R[i] = ret

        R = torch.from_numpy(np.asarray(R)).float().to(args.device)
        GAE = torch.from_numpy(np.asarray(GAE)).float().to(args.device)
        values = torch.stack(values[:-1])
        entropies = torch.stack(entropies, dim=-1)
        log_probs = torch.stack(log_probs, dim=-1)

        assert values.shape == R.shape, (
            values.shape,
            R.shape,
        )

        Advs = R - values

        with lock:
            optimizer.zero_grad()

            assert Advs.requires_grad
            value_loss = 0.5 * Advs.pow(2).mean()

            assert log_probs.shape == GAE.shape, (
                log_probs.shape,
                GAE.shape,
            )
            assert log_probs.requires_grad

            pg_loss = -(log_probs * GAE.detach()).mean()
            entropy_loss = entropies.mean()

            total_loss = (
                pg_loss
                + args.value_loss_coef * value_loss
                - args.entropy_coef * entropy_loss
            )

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                agent.model.parameters(), args.max_grad_norm
            )
            optimizer.step()

        with lock:
            writer.add_scalar(
                "training/entropy" + str(rank), entropy_loss.item(), epoch
            )
            writer.add_scalar(
                "training/value_loss" + str(rank), value_loss.item(), epoch
            )
            writer.add_scalar("training/pg_loss" + str(rank), pg_loss.item(), epoch)
            writer.add_scalar(
                "training/total_loss" + str(rank), total_loss.item(), epoch
            )
            writer.add_scalar(
                "training/grad_norm" + str(rank), grad_norm.cpu().item(), epoch
            )
            writer.add_scalar(
                "training/adv" + str(rank), Advs.detach().mean().item(), epoch
            )
            writer.add_scalar(
                "training/gae" + str(rank), GAE.detach().mean().item(), epoch
            )

        if not args.async_mode:
            rollout(
                epoch,
                counter,
                lock,
                agent,
                shared_model,
                env,
                start_time,
                writer,
                args,
            )
