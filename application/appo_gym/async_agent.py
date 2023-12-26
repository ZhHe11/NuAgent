from typing import Any, Dict, Tuple

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch._tensor import Tensor

from torch.distributions import Categorical
from uniagent.core.agent_runner import EpisodeState

from application.a3c_gym.async_agent import (
    AsyncAgent as BaseRunner,
    compute_gae_and_ret,
)


class AsyncAgent(BaseRunner):
    def compute_loss(
        self, episode_state: EpisodeState
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        episode_state = compute_gae_and_ret(self.args, episode_state)
        old_log_prob = episode_state.log_probs.detach()
        actions = (
            torch.from_numpy(np.concatenate(episode_state.actions))
            .long()
            .to(self.device)
        )

        Adv = episode_state.adv
        for _ in range(self.args.repeat):
            if self.args.recompute_adv:
                episode_state = compute_gae_and_ret(self.args, episode_state)

            if self.args.norm_adv:
                mean, std = Adv.mean(), Adv.std()
                Adv = (Adv - mean) / (std + 1e-8)

            dist = Categorical(logits=episode_state.logits)
            log_prob = dist.log_prob(actions)

            ratio = torch.exp(log_prob - old_log_prob).float()
            surr1 = ratio * Adv.detach()
            surr2 = (
                torch.clamp(ratio, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip)
                * Adv.detach()
            )

            if self.args.dual_clip > 0:
                clip1 = torch.min(surr1, surr2)
                clip2 = torch.max(surr1, self.args.dual_clip * Adv)
                clip_loss = -torch.where(Adv < 0, clip2, clip1).mean()
            else:
                clip_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = dist.entropy().mean()
            loss = (
                clip_loss
                + self.args.value_loss_coef * Adv.pow(2).mean()
                - self.args.entropy_coef * entropy_loss
            )
            return loss, {
                "pg_loss": clip_loss.item(),
                "adv": Adv.mean().item(),
                "value_loss": Adv.mean().item(),
                "entropy_loss": entropy_loss.item(),
                "total_loss": loss.item(),
            }

    def log_training(
        self, epoch: int, loss_detail: Dict[str, Tensor], writer: SummaryWriter
    ):
        writer.add_scalar(
            "training/entropy" + str(self.rank),
            loss_detail["entropy_loss"].item(),
            epoch,
        )
        writer.add_scalar(
            "training/value_loss" + str(self.rank),
            loss_detail["value_loss"].item(),
            epoch,
        )
        writer.add_scalar(
            "training/pg_loss" + str(self.rank), loss_detail["pg_loss"].item(), epoch
        )
        writer.add_scalar(
            "training/adv" + str(self.rank), loss_detail["adv"].item(), epoch
        )
        writer.add_scalar(
            "training/total_loss" + str(self.rank),
            loss_detail["total_loss"].item(),
            epoch,
        )
