from typing import Any, Dict, Tuple
from itertools import count

import threading
import numpy as np
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from torch import optim
from torch.distributions import Categorical
from torch.nn.modules import Module

from uniagent.trainers.parameter_server import ParameterServer
from uniagent.models.fun.feudal_net import FeudalState, FeudalNet
from application.a3c_gym.async_agent import AsyncAgent as BaseAgent, EpisodeState


class FeudalParameterServer(ParameterServer):
    def setup_optimizer(self):
        optimizer = {}
        if self.args.optimizer == "sgd":
            optimizer["manager"] = optim.SGD(
                self.model.manager.parameters(), lr=self.args.manager_lr, momentum=0.9
            )
            optimizer["worker"] = optim.SGD(
                self.model.worker.parameters(), lr=self.args.worker_lr, momentum=0.9
            )
        elif self.args.optimizer == "rmsprop":
            optimizer["manager"] = optim.RMSprop(
                self.model.manager.parameters(), lr=self.args.manager_lr
            )
            optimizer["worker"] = optim.RMSprop(
                self.model.manager.parameters(), lr=self.args.worker_lr
            )
        elif self.args.optimizer == "adam":
            optimizer["manager"] = optim.Adam(
                self.model.worker.parameters(), lr=self.args.manager_lr
            )
            optimizer["worker"] = optim.Adam(
                self.model.worker.parameters(), lr=self.args.worker_lr
            )
        else:
            raise NotImplementedError
        return optimizer

    def step_optimizer(self):
        for v in self.optimizer.values():
            v.step()
            v.zero_grad()


from numbers import Number


class AsyncAgent(BaseAgent):
    def fetch_model(self) -> Module:
        model_cnt, model = self.ps_rref.rpc_sync().get_model()
        return model.to(self.device)

    def update_and_fetch_model(self, model: Module) -> Module:
        for p in model.parameters():
            if p.grad is None:
                raise RuntimeError(
                    f"Empty grad at worker: {self.worker_name} {self.rank}"
                )

        model: nn.Module = rpc.rpc_sync(
            self.ps_rref.owner(),
            FeudalParameterServer.update_and_fetch_model,
            args=(
                self.ps_rref,
                self.worker_name,
                [p.grad for p in model.cpu().parameters()],
            ),
            timeout=60,
        ).to(self.device)
        return model

    def run_episode(
        self,
        obs: np.ndarray,
        last_done: bool,
        net_state: FeudalState,
        global_counter: mp.Value = None,
        lock: threading.Lock = None,
    ) -> EpisodeState:
        rewards = []
        intrinsic_rewards = []
        values_manager = []
        values_worker = []
        # net_states_manager = []
        # net_states_worker = []

        log_probs = []
        entropies = []
        obses = []
        actions = []
        dones = []

        assert isinstance(self.model, FeudalNet)

        if last_done:
            net_state: FeudalState = self.model.init_state(1, self.device)
        else:
            net_state = self.model.reset_states_grad(net_state)

        counter = count() if not self.model.training else range(self.args.num_steps)
        # counter = range(self.args.num_steps)

        for step_cnt in counter:
            obses.append(obs)
            # TODO(ming): fix this squeeze
            # net_states_worker.append(
            #     tuple(map(lambda x: x.squeeze(0), net_state.worker_state))
            # )
            # # tick, hx, cx
            # net_states_manager.append(
            #     tuple(map(lambda x: x.squeeze(0) if not isinstance(x, Number) else x, net_state.manager_state))
            # )
            # net_states_worker.append(net_state.worker_state)
            # net_states_manager.append(net_state.manager_state)

            obs = torch.from_numpy(obs).float().to(self.device)
            value_worker, value_manager, action_logits, net_state = self.model(
                obs.unsqueeze(0), net_state, flush_network_state=True
            )

            dist = Categorical(logits=action_logits)

            if self.model.training:
                # XXX(ming): only discreate distribution use sample
                #   otherwise rsample
                action = dist.sample()
            else:
                action = action_logits.argmax(dim=-1)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            obs, reward, done, truncated, info = self.env.step(action.cpu().numpy()[0])
            done = done or truncated
            intrinsic_reward = (
                self.model.intrinsic_reward(obs, reward, info, net_state).cpu().item()
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
        # net_states_worker.append(
        #     tuple(map(lambda x: x.squeeze(0), net_state.worker_state))
        # )
        # net_states_manager.append(
        #     tuple(map(lambda x: x.squeeze(0), net_state.worker_state))
        # )

        if dones[-1]:
            values_manager.append(torch.zeros(1).to(self.device).squeeze())
            values_worker.append(torch.zeros(1).to(self.device).squeeze())
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
            value_worker, value_manager, _, _ = self.model(
                obs.unsqueeze(0), net_state, flush_network_state=False
            )
            values_manager.append(value_manager.squeeze())
            values_worker.append(value_worker.squeeze())

        return EpisodeState(
            obses=obses,
            dones=dones,
            actions=actions,
            # net_states_manager=net_states_manager,
            # net_states_worker=net_states_worker,
            last_net_state=net_state,
            intrinsic_rewards=intrinsic_rewards,
            rewards=rewards,
            values_manager=values_manager,
            values_worker=values_worker,
            worker_log_probs=log_probs,
            worker_entropies=entropies,
            episode_len=len(rewards),
        )

    def handle_net_states(self, episode_state: EpisodeState) -> Any:
        return episode_state.last_net_state

    def compute_loss(
        self, episode_state: EpisodeState
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rewards = episode_state.rewards
        intrinsic_rewards = episode_state.intrinsic_rewards
        values_manager = episode_state.values_manager
        values_worker = episode_state.values_worker
        goal_hist = episode_state.last_net_state.goal_seg
        state_hist = episode_state.last_net_state.state_seg

        assert isinstance(self.model, FeudalNet)
        assert (
            len(state_hist)
            == len(goal_hist)
            == episode_state.episode_len + self.model.config.c
        ), (
            len(state_hist),
            len(goal_hist),
            episode_state.episode_len + self.model.config.c,
        )

        # in fact, we use cos_similarity to approximate the
        #   goal generation policy, which is optimized towards
        #   directed state vector (pointed from s_t to s_{t+c})
        dcos_ts = self.model.state_cosin_similarity(
            state_hist, goal_hist, use_repeated_terminal_state=True
        )

        gae_worker = 0.0
        gae_manager = 0.0
        ret_worker = values_worker[-1].item()
        ret_manager = values_manager[-1].item()
        R_worker = [0.0] * episode_state.episode_len
        R_manager = [0.0] * episode_state.episode_len
        GAE_manager = [0.0] * episode_state.episode_len
        GAE_worker = [0.0] * episode_state.episode_len

        for i in range(episode_state.episode_len - 1, -1, -1):
            # for worker
            ret_worker = (
                self.args.gamma_worker * ret_worker
                + rewards[i]
                + self.args.alpha * intrinsic_rewards[i]
            )
            delta_t_worker = (
                rewards[i]
                + self.args.alpha * intrinsic_rewards[i]
                + self.args.gamma_worker * values_worker[i + 1].cpu().item()
                - values_worker[i].cpu().item()
            )
            gae_worker = (
                gae_worker * self.args.gamma_worker * self.args.lambda_worker
                + delta_t_worker
            )

            # for manager
            ret_manager = self.args.gamma_manager * ret_manager + rewards[i]
            delta_t_manager = (
                rewards[i]
                + self.args.gamma_manager * values_manager[i + 1].cpu().item()
                - values_manager[i].cpu().item()
            )
            gae_manager = (
                gae_manager * self.args.gamma_manager * self.args.lambda_manager
                + delta_t_manager
            )

            GAE_worker[i] = gae_worker
            GAE_manager[i] = gae_manager
            R_worker[i] = ret_worker
            R_manager[i] = ret_manager

        R_worker = torch.from_numpy(np.asarray(R_worker)).float().to(self.device)
        R_manager = torch.from_numpy(np.asarray(R_manager)).float().to(self.device)
        GAE_manager = torch.from_numpy(np.asarray(GAE_manager)).float().to(self.device)
        GAE_worker = torch.from_numpy(np.asarray(GAE_worker)).float().to(self.device)

        values_manager = torch.stack(values_manager[:-1])
        values_worker = torch.stack(values_worker[:-1])
        entropies = torch.stack(episode_state.worker_entropies, dim=-1)
        log_probs = torch.stack(episode_state.worker_log_probs, dim=-1)

        assert values_worker.shape == R_worker.shape, (
            values_worker.shape,
            R_worker.shape,
        )

        # Advs_worker = R_worker - values_worker
        Advs_worker = GAE_worker - values_worker

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
        manager_pg_loss = (dcos_ts * GAE_manager.detach()).mean()

        entropy_loss = entropies.mean()

        worker_total_loss = (
            worker_pg_loss
            + self.args.value_worker_loss_coef * worker_value_loss
            - self.args.entropy_coef * entropy_loss
        )
        manager_total_loss = (
            manager_pg_loss + self.args.value_manager_loss_coef * manager_value_loss
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
        self, epoch: int, loss_detail: Dict[str, torch.Tensor], writer: SummaryWriter
    ):
        loss_detail["training/info"]["grad_norm"] = loss_detail.pop("grad_norm")

        for k, v in loss_detail.items():
            writer.add_scalars(k + str(self.rank), v, epoch)
