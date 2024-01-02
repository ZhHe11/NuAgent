from typing import Any, Dict, Tuple
from argparse import Namespace
from itertools import count

import time
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from torch.distributions import Categorical
from torch.nn.modules import Module
from tensorboardX import SummaryWriter

from uniagent.core.agent_runner import EpisodeState
from uniagent.utils.statistics import RunningMeanStd
from uniagent.trainers.parameter_server import ParameterServer, setup_optimizer

from application.a3c_gym.async_agent import (
    AsyncAgent as BaseRunner,
    compute_gae_and_ret,
)


class AsyncAgent(BaseRunner):
    def __init__(
        self,
        args: Namespace,
        ps_rref: Any,
        rank: int,
        model_class: Module,
        model_kwargs: dict,
        make_env: Any,
        log_dir: str,
    ) -> None:
        super().__init__(
            args, ps_rref, rank, model_class, model_kwargs, make_env, log_dir
        )
        self.ret_rms = RunningMeanStd()
        self.grad_placeholder = []

    def save_grad(self):
        i = 0
        for p in self.model.parameters():
            if p.grad is not None:
                if len(self.grad_placeholder) <= i + 1:
                    self.grad_placeholder.append(torch.zeros_like(p.grad))
                self.grad_placeholder[i] += p.grad.data.clone()
                i += 1

    def update_and_fetch_model(self, model: Module) -> Module:
        model: nn.Module = rpc.rpc_sync(
            self.ps_rref.owner(),
            ParameterServer.update_and_fetch_model,
            args=(
                self.ps_rref,
                self.worker_name,
                self.grad_placeholder,
            ),
        ).to(self.device)
        model.train()
        self.grad_placeholder = []
        return model

    def compute_loss(
        self, episode_state: EpisodeState
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        optimizer = setup_optimizer(self.args, self.model)
        episode_state = self.preprocess_episode(episode_state)
        episode_state = compute_gae_and_ret(self.args, self.model, episode_state)

        pg_losses = []
        advs = []
        value_losses = []
        entropy_losses = []
        losses = []

        for _ in range(self.args.repeat):
            if self.args.recompute_adv:
                episode_state = compute_gae_and_ret(
                    self.args, self.model, episode_state, recompute_value=True
                )

            for minibatch in episode_state.split(self.args.batch_size):
                if minibatch.episode_len == 0:
                    continue
                Adv = minibatch.gae
                old_log_prob = minibatch.old_log_probs
                R = minibatch.rets

                if self.args.norm_adv:
                    mean, std = Adv.mean(), Adv.std()
                    Adv = (Adv - mean) / (std + 1e-8)

                values, logits, _ = self.model(
                    minibatch.obses,
                    minibatch.net_states,
                )
                values = values.squeeze(-1)
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(minibatch.actions)
                entropy_loss = dist.entropy().mean()

                assert log_prob.shape == old_log_prob.shape, (
                    log_prob.shape,
                    old_log_prob.shape,
                )
                ratio = torch.exp(log_prob - old_log_prob.detach()).float()

                assert ratio.shape == Adv.shape, (ratio.shape, Adv.shape)
                surr1 = ratio * Adv.detach()
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip
                    )
                    * Adv.detach()
                )
                assert len(surr1.shape) == 1

                if self.args.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(surr1, self.args.dual_clip * Adv)
                    clip_loss = -torch.where(Adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                if self.args.value_clip:
                    assert minibatch.state_values.shape == v_clip.shape, (
                        minibatch.state_values.shape,
                        v_clip.shape,
                    )
                    v_clip = minibatch.state_values + (
                        values - minibatch.state_values
                    ).clamp(-self.args.eps_clip, self.args.eps_clip)
                    assert minibatch.rets.shape == values.shape == v_clip.shape, (
                        minibatch.rets.shape,
                        values.shape,
                        v_clip.shape,
                    )
                    vf1 = (minibatch.rets - values).pow(2)
                    vf2 = (minibatch.rets - v_clip).pow(2)
                    value_loss = 0.5 * torch.max(vf1, vf2).mean()
                else:
                    assert R.shape == values.shape, (R.shape, values.shape)
                    value_loss = 0.5 * (R.detach() - values).pow(2).mean()

                loss = (
                    clip_loss
                    + self.args.value_loss_coef * value_loss
                    - self.args.entropy_coef * entropy_loss
                )
                optimizer.zero_grad()
                loss.backward()
                self.save_grad()
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )
                    .cpu()
                    .item()
                )
                optimizer.step()
                losses.append(loss.item())
                pg_losses.append(clip_loss.item())
                advs.append(Adv.mean().item())
                value_losses.append(value_loss.mean().item())
                entropy_losses.append(entropy_loss.item())

        return {
            "grad_norm": grad_norm,
            "pg_loss": sum(pg_losses) / len(pg_losses),
            "adv": sum(advs) / len(advs),
            "value_loss": sum(value_losses) / len(value_losses),
            "entropy_loss": sum(entropy_losses) / len(entropy_losses),
            "total_loss": sum(losses) / len(losses),
        }

    def log_training(
        self, epoch: int, loss_detail: Dict[str, torch.Tensor], writer: SummaryWriter
    ):
        writer.add_scalar(
            "training/entropy" + str(self.rank),
            loss_detail["entropy_loss"],
            epoch,
        )
        writer.add_scalar(
            "training/value_loss" + str(self.rank),
            loss_detail["value_loss"],
            epoch,
        )
        writer.add_scalar(
            "training/pg_loss" + str(self.rank), loss_detail["pg_loss"], epoch
        )
        writer.add_scalar("training/adv" + str(self.rank), loss_detail["adv"], epoch)
        writer.add_scalar(
            "training/total_loss" + str(self.rank),
            loss_detail["total_loss"],
            epoch,
        )
        writer.add_scalar(
            "training/grad_norm" + str(self.rank), loss_detail["grad_norm"], epoch
        )

    def train(self, counter: mp.Value = None, lock: threading.Lock = None):
        self.model: nn.Module = self.fetch_model()
        time.sleep(1)
        writer = SummaryWriter(log_dir=self.log_dir)

        obs, _ = self.env.reset()
        last_done = True
        last_net_states = None

        for epoch in count():
            self.model.train()
            episode_state = self.run_episode(
                obs, last_done, last_net_states, counter, lock
            )

            with lock:
                writer.add_scalars(
                    "training/episode_info" + str(self.rank),
                    {
                        "episode_reward": sum(episode_state.rewards),
                        "episode_length": episode_state.episode_len,
                    },
                    epoch,
                )

            loss_detail = self.compute_loss(episode_state)

            # then sync model
            self.model = self.update_and_fetch_model(self.model)
            assert self.model.training

            with lock:
                self.log_training(epoch, loss_detail, writer)

            if episode_state.dones[-1]:
                obs, _ = self.env.reset()
            else:
                obs = episode_state.obses[-1]
            last_done = episode_state.dones[-1]
            last_net_states = self.handle_net_states(episode_state)
