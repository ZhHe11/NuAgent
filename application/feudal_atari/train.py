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


# def feudal_loss(obs, rewards, next_v_M, next_v_W, args: Namespace):
#     R_M = next_v_M
#     R_W = next_v_W

#     traj_len = len(rewards)

#     for i in reversed(range(traj_len)):
#         R_M = rewards[i] + args.gamma_manager * R_M
#         R_W = rewards[i] + args.gamma_worker * R_W


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

        # if done:
        states: FeudalState = model.init_state(1)
        # else:
        #     states: FeudalState = model.reset_states_grad(states)

        values_worker, values_manager = [], []
        log_probs = []
        rewards, intrinsic_rewards = [], []
        entropies = []  # regularisation
        dcos_t_minus_cs = []

        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(args.device)
        done = True

        for step in range(args.num_steps):
            episode_length += 1
            (
                value_worker,
                value_manager,
                action_probs,
                goal,
                dcos_t_minus_c,
                states,
            ) = model(obs.unsqueeze(0), states, reset_value_grad=False)

            m = Categorical(probs=action_probs)
            action = m.sample()
            log_prob = m.log_prob(action).squeeze()
            entropy = m.entropy().squeeze()
            entropies.append(entropy)
            dcos_t_minus_cs.append(dcos_t_minus_c.squeeze())

            next_obs, reward, done, truncated, info = env.step(action.cpu().numpy()[0])
            done = done or truncated
            intrinsic_reward = (
                model.intrinsic_reward(obs, reward, info, states).cpu().item()
            )

            with lock:
                counter.value += 1

            # if done:
            #     next_obs, _ = env.reset()

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
        # print(f"[training] done for step: {step}, trucated? {truncated} done? {done}")

        # if last is not done, bootstrap value target
        if not done:
            value_worker, value_manager, _, _, _, _ = model(obs.unsqueeze(0), states)
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

        # gae_worker = torch.zeros(1).to(args.device).squeeze()
        traj_len = len(rewards)
        ret_worker = values_worker[-1].item()
        ret_manager = values_manager[-1].item()
        R_worker = [0.0] * traj_len
        R_manager = [0.0] * traj_len

        for i in reversed(range(traj_len)):
            ret_worker = (
                args.gamma_worker * ret_worker
                + rewards[i]
                + args.alpha * intrinsic_rewards[i]
            )
            ret_manager = args.gamma_manager * ret_manager + rewards[i]
            R_worker[i] = ret_worker
            R_manager[i] = ret_manager

        R_worker = torch.from_numpy(np.asarray(R_worker)).float().to(args.device)
        R_manager = torch.from_numpy(np.asarray(R_manager)).float().to(args.device)
        values_manager = torch.stack(values_manager[:-1])
        values_worker = torch.stack(values_worker[:-1])
        entropies = torch.stack(entropies, dim=-1)
        log_probs = torch.stack(log_probs, dim=-1)
        dcos_t_minus_cs = torch.stack(dcos_t_minus_cs)
        dcos_ts = torch.concat(
            [
                dcos_t_minus_cs,
                torch.autograd.Variable(
                    torch.zeros(
                        (model.manager.c,) + dcos_t_minus_cs.shape[1:],
                        dtype=dcos_t_minus_cs.dtype,
                        device=dcos_t_minus_cs.device,
                    ),
                    requires_grad=True,
                ),
            ],
            dim=0,
        )[model.manager.c :]
        # print(f"--------- shapes of values_manager: {values_manager.shape}, values_worker: {values_worker.shape}, entropies: {entropies.shape}, log_probs: {log_probs.shape}, dcos_t_minus_cs: {dcos_t_minus_cs.shape}")

        assert values_worker.shape == R_worker.shape, (
            values_worker.shape,
            R_worker.shape,
        )

        Advs_worker = R_worker - values_worker

        assert values_manager.shape == R_manager.shape, (
            values_manager.shape,
            R_manager.shape,
        )
        Advs_manager = R_manager - values_manager

        optimizer.zero_grad()

        assert Advs_manager.requires_grad and Advs_worker.requires_grad, (
            Advs_worker.requires_grad,
            Advs_manager.requires_grad,
        )
        worker_value_loss = 0.5 * Advs_worker.pow(2).mean()
        manager_value_loss = 0.5 * Advs_manager.pow(2).mean()

        assert log_probs.shape == Advs_worker.shape, (
            log_probs.shape,
            Advs_worker.shape,
        )
        assert log_probs.requires_grad
        worker_pg_loss = -(log_probs * Advs_worker.detach()).mean()
        assert dcos_ts.shape == Advs_manager.shape, (
            dcos_ts.shape,
            Advs_manager.shape,
        )
        assert dcos_ts.requires_grad
        manager_pg_loss = -(dcos_ts * Advs_manager.detach()).mean()

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

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        # value_worker_loss = value_worker_loss + 0.5 * advantage_worker.pow(2)
        # value_manager_loss = value_manager_loss + 0.5 * advantage_manager.pow(2)

        #     # Generalized Advantage Estimation
        #     delta_t_worker = (
        #         rewards[i]
        #         + args.alpha * intrinsic_rewards[i]
        #         + args.gamma_worker * values_worker[i + 1].data
        #         - values_worker[i].data
        #     )
        #     gae_worker = (
        #         gae_worker * args.gamma_worker * args.tau_worker + delta_t_worker
        #     )

        #     assert log_probs[i].size() == entropies[i].size() == gae_worker.size(), (
        #         log_probs[i].size(),
        #         entropies[i].size(),
        #         gae_worker.size(),
        #     )

        #     policy_loss = (
        #         policy_loss
        #         - log_probs[i] * gae_worker.detach()
        #         - args.entropy_coef * entropies[i]
        #     )

        #     if (i + model.c) < traj_len:
        #         # TODO try padding the manager_partial_loss with end values (or zeros)
        #         assert (
        #             advantage_manager.shape == manager_partial_loss[i + model.c].shape
        #         ), (advantage_manager.shape, manager_partial_loss[i + model.c].shape)
        #         manager_loss = (
        #             manager_loss
        #             - advantage_manager.detach() * manager_partial_loss[i + model.c]
        #         )

        # if traj_len == 0:
        #     continue
