from itertools import count
from argparse import Namespace

import numpy as np
import torch
import torch.optim as optim

from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

from uniagent.models.fun import FeudalNet
from uniagent.models.fun.feudal_net import FeudalState
from uniagent.envs.atari import create_atari_env

from tensorboardX import SummaryWriter


def ensure_shared_grads(model: nn.Module, shared_model: nn.Module):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
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

    env = create_atari_env(args.env_name, args.max_episode_length)
    env.seed(seed)
    model = FeudalNet(
        env.observation_space,
        env.action_space,
        channel_first=args.channel_first,
        device=args.device,
    )

    if optimizer is None:
        print("no shared optimizer")
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=log_dir)

    model.train()

    episode_length = 0
    for epoch in count():
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        states: FeudalState = model.init_state(1)

        values_worker, values_manager = [], []
        log_probs = []
        rewards, intrinsic_rewards = [], []
        entropies = []  # regularisation

        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(args.device)
        done = True

        for step in range(args.num_steps):
            episode_length += 1
            (
                value_worker,
                value_manager,
                action_probs,
                states,
            ) = model(obs.unsqueeze(0), states, reset_value_grad=False)

            m = Categorical(probs=action_probs)
            action = m.sample()
            log_prob = m.log_prob(action).squeeze()
            entropy = m.entropy().squeeze()
            entropies.append(entropy)

            next_obs, reward, done, truncated, info = env.step(action.cpu().numpy()[0])
            done = done or truncated
            intrinsic_reward = (
                model.intrinsic_reward(obs, reward, info, states).cpu().item()
            )

            with lock:
                counter.value += 1

            values_manager.append(value_manager.squeeze())
            values_worker.append(value_worker.squeeze())
            log_probs.append(log_prob)
            rewards.append(reward)
            intrinsic_rewards.append(intrinsic_reward)

            if done:
                break

            obs = torch.from_numpy(next_obs).to(args.device)

        if step < model.c:
            continue

        if not done:
            value_worker, value_manager, _, _ = model(
                obs.unsqueeze(0), states, update_network_state=False
            )
            value_worker = value_worker.detach().squeeze()
            value_manager = value_manager.detach().squeeze()
        else:
            value_worker = torch.zeros(1).to(args.device).squeeze()
            value_manager = torch.zeros(1).to(args.device).squeeze()

        with lock:
            writer.add_scalars(
                "training/episode_info" + str(rank),
                {
                    "episode_reward": sum(rewards),
                    "intinsic_reward": sum(intrinsic_rewards),
                    "episode_length": len(rewards),
                },
                epoch,
            )

        values_worker.append(value_worker)
        values_manager.append(value_worker)

        gae_worker = 0.0  # torch.zeros(1).to(args.device).squeeze()
        gae_manager = 0.0  # torch.zeros(1).to(args.device).squeeze()
        traj_len = len(rewards)
        ret_worker = values_worker[-1].item()
        ret_manager = values_manager[-1].item()
        R_worker = [0.0] * traj_len
        R_manager = [0.0] * traj_len
        GAE_manager = [0.0] * traj_len
        GAE_worker = [0.0] * traj_len

        for i in reversed(range(traj_len)):
            ret_worker = (
                args.gamma_worker * ret_worker
                + rewards[i]
                + args.alpha * intrinsic_rewards[i]
            )
            delta_t_worker = (
                rewards[i]
                + args.alpha * intrinsic_rewards[i]
                + args.gamma_worker * values_worker[i + 1].cpu().item()
                - values_worker[i].cpu().item()
            )
            gae_worker = (
                gae_worker * args.gamma_worker * args.lambda_worker + delta_t_worker
            )
            ret_manager = args.gamma_manager * ret_manager + rewards[i]
            delta_t_manager = (
                rewards[i]
                + args.gamma_manager * values_manager[i + 1].cpu().item()
                - values_manager[i].cpu().item()
            )
            gae_manager = (
                gae_manager * args.gamma_manager * args.lambda_manager + delta_t_manager
            )

            GAE_worker[i] = gae_worker
            GAE_manager[i] = gae_manager
            R_worker[i] = ret_worker
            R_manager[i] = ret_manager

        R_worker = torch.from_numpy(np.asarray(R_worker)).float().to(args.device)
        R_manager = torch.from_numpy(np.asarray(R_manager)).float().to(args.device)
        GAE_manager = torch.from_numpy(np.asarray(GAE_manager)).float().to(args.device)
        GAE_worker = torch.from_numpy(np.asarray(GAE_worker)).float().to(args.device)
        values_manager = torch.stack(values_manager[:-1])
        values_worker = torch.stack(values_worker[:-1])
        entropies = torch.stack(entropies, dim=-1)
        log_probs = torch.stack(log_probs, dim=-1)
        dcos_ts = model.state_cosin_similarity(
            states.state_seg, states.goal_seg, use_repeated_terminal_state=True
        )

        assert values_worker.shape == R_worker.shape, (
            values_worker.shape,
            R_worker.shape,
        )

        Advs_worker = R_worker - values_worker
        # Advs_worker = GAE_worker - values_worker

        assert values_manager.shape == R_manager.shape, (
            values_manager.shape,
            R_manager.shape,
        )
        Advs_manager = R_manager - values_manager
        # Advs_manager = GAE_manager - values_manager

        optimizer.zero_grad()

        assert Advs_manager.requires_grad and Advs_worker.requires_grad, (
            Advs_worker.requires_grad,
            Advs_manager.requires_grad,
        )
        worker_value_loss = 0.5 * Advs_worker.pow(2).mean()
        manager_value_loss = 0.5 * Advs_manager.pow(2).mean()

        assert log_probs.shape == GAE_worker.shape, (
            log_probs.shape,
            GAE_manager.shape,
        )
        assert log_probs.requires_grad
        worker_pg_loss = -(log_probs * GAE_worker.detach()).mean()
        assert dcos_ts.shape == GAE_manager.shape, (
            dcos_ts.shape,
            GAE_manager.shape,
        )
        assert dcos_ts.requires_grad
        manager_pg_loss = -(dcos_ts * GAE_manager.detach()).mean()

        entropy_loss = entropies.mean()

        worker_total_loss = (
            worker_pg_loss
            + args.value_worker_loss_coef * worker_value_loss
            - args.entropy_coef * entropy_loss
        )
        manager_total_loss = (
            manager_pg_loss + args.value_manager_loss_coef * manager_value_loss
        )

        total_loss = worker_total_loss + manager_total_loss

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )

        with lock:
            writer.add_scalars(
                "training/info" + str(rank),
                {
                    "entropy": entropy_loss.item(),
                    "grad_norm": grad_norm.cpu().item(),
                },
                epoch,
            )
            writer.add_scalars(
                "training/advantages" + str(rank),
                {
                    "manager": Advs_manager.detach().mean().item(),
                    "worker": Advs_worker.detach().mean().item(),
                },
                epoch,
            )
            writer.add_scalars(
                "training/pg_loss" + str(rank),
                {
                    "manager": manager_pg_loss.item(),
                    "worker": worker_pg_loss.item(),
                },
                epoch,
            )
            writer.add_scalars(
                "training/total_loss",
                {
                    "manager": manager_total_loss.item(),
                    "worker": worker_total_loss.item(),
                },
                epoch,
            )
            writer.add_scalars(
                "training/value_loss" + str(rank),
                {
                    "worker": worker_value_loss.item(),
                    "manager": manager_value_loss.item(),
                },
                epoch,
            )
            writer.add_scalars(
                "training/pg_loss_detail" + str(rank),
                {
                    "cosin_similarity": dcos_ts.detach().mean().item(),
                    "gae_manager": GAE_manager.detach().mean().item(),
                    "gae_worker": GAE_worker.detach().mean().item(),
                },
                epoch,
            )

        ensure_shared_grads(model, shared_model)
        optimizer.step()
