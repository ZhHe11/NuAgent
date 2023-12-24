from multiprocessing import Value
from typing import Any, Dict, Sequence, Tuple
from argparse import Namespace
from itertools import count

import threading
from tensorboardX import SummaryWriter
import torch
import numpy as np
from torch._tensor import Tensor
import torch.nn as nn
import torch.multiprocessing as mp

from torch.distributions import Categorical
from torch.nn.modules import Module
from application.a3c_gym.async_agent import AsyncAgent as BaseAgent, EpisodeState
from uniagent.models.fun.feudal_net import FeudalState


class AsyncAgent(BaseAgent):
    def run_episode(
        self,
        args: Namespace,
        obs: np.ndarray,
        model: nn.Module,
        global_counter: mp.Value = None,
        lock: threading.Lock = None,
    ) -> Sequence[Any]:
        done = False

        rewards = []
        intrinsic_rewards = []
        values_manager = []
        values_worker = []
        net_states_manager = []
        net_states_worker = []

        log_probs = []
        entropies = []
        obses = []
        actions = []
        dones = []

        net_state: FeudalState = model.init_state(1, self.device)
        counter = count() if model.training else range(args.num_steps)

        for step_cnt in counter:
            obses.append(obs)
            # TODO(ming): fix this squeeze
            net_states_worker.append(
                tuple(map(lambda x: x.squeeze(0), net_state.worker_state))
            )
            net_states_manager.append(
                tuple(map(lambda x: x.squeeze(0), net_state.worker_state))
            )

            obs = torch.from_numpy(obs).float()
            value_worker, value_manager, action_probs, net_state = self.model(
                obs.unsqueeze(0), net_state
            )

            dist = Categorical(probs=action_probs)
            if self.model.training:
                action = dist.sample()
            else:
                action = action_probs.argmax(dim=-1)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            obs, reward, done, truncated, info = self.env.step(action.cpu().numpy()[0])
            done = done or truncated
            intrinsic_reward = (
                model.intrinsic_reward(obs, reward, info, net_state).cpu().item()
            )

            values_worker.append(value_worker.squeeze())
            values_manager.append(value_manager.squeeze())
            actions.append(action)
            log_probs.append(log_prob.squeeze())
            entropies.append(entropy.squeeze())
            rewards.append(reward)
            intrinsic_rewards.append(intrinsic_reward)
            dones.append(done)

            if global_counter:
                with lock:
                    global_counter.value += 1

            if done:
                break

        obses.append(obs)
        net_states_worker.append(
            tuple(map(lambda x: x.squeeze(0), net_state.worker_state))
        )
        net_states_manager.append(
            tuple(map(lambda x: x.squeeze(0), net_state.worker_state))
        )

        if dones[-1]:
            values_manager.append(torch.zeros(1).to(self.device).squeeze())
            values_worker.append(torch.zeros(1).to(self.device).squeeze())
        else:
            obs = torch.from_numpy(obs).float()
            value_worker, value_manager, _, _ = model(obs.unsqueeze(0), net_state)

        return EpisodeState(
            obses,
            dones,
            actions,
            [net_states_manager, net_states_worker, net_state, intrinsic_rewards],
            rewards,
            [values_manager, values_worker],
            log_probs,
            entropies,
            len(rewards),
        )

    def compute_loss(
        self, args: Namespace, model: Module, episode_state: EpisodeState
    ) -> Tuple[Tensor, Dict[str, Any]]:
        rewards = episode_state.rewards
        intrinsic_rewards = episode_state.net_states[-1]
        values_manager = episode_state.values[0]
        values_worker = episode_state.values[1]
        states = episode_state.net_states[2]

        gae_worker = 0.0  # torch.zeros(1).to(args.device).squeeze()
        gae_manager = 0.0  # torch.zeros(1).to(args.device).squeeze()
        ret_worker = values_worker[-1].item()
        ret_manager = values_manager[-1].item()
        R_worker = [0.0] * episode_state.episode_len
        R_manager = [0.0] * episode_state.episode_len
        GAE_manager = [0.0] * episode_state.episode_len
        GAE_worker = [0.0] * episode_state.episode_len

        for i in reversed(range(episode_state.episode_len)):
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
        entropies = torch.stack(episode_state.entropies, dim=-1)
        log_probs = torch.stack(episode_state.log_probs, dim=-1)
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

        return total_loss, {
            "training/info": {
                "entropy": entropy_loss.item(),
            },
            "training/advantages": {
                "manager": Advs_manager.detach().mean().item(),
                "worker": Advs_worker.detach().mean().item(),
            },
            "training/pg_loss": {
                "manager": manager_pg_loss.item(),
                "worker": worker_pg_loss.item(),
            },
            "training/total_loss": {
                "manager": manager_total_loss.item(),
                "worker": worker_total_loss.item(),
            },
            "training/value_loss": {
                "worker": worker_value_loss.item(),
                "manager": manager_value_loss.item(),
            },
            "training/pg_loss_detail": {
                "cosin_similarity": dcos_ts.detach().mean().item(),
                "gae_manager": GAE_manager.detach().mean().item(),
                "gae_worker": GAE_worker.detach().mean().item(),
            },
        }

    def log_training(
        self, epoch: int, loss_detail: Dict[str, Tensor], writer: SummaryWriter
    ):
        loss_detail["training/info"]["grad_norm"] = loss_detail.pop("grad_norm")

        for k, v in loss_detail.items():
            writer.add_scalars(k + str(self.rank), v, epoch)
