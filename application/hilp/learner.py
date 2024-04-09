from typing import Tuple, Dict, Any, Sequence, Union

import copy

from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch

from torch import nn

from .networks import (
    GoalConditionedPhiValue,
    GoalConditionedValue,
    GoalConditionedCritic,
    Actor,
)


def expectile_loss(adv: torch.Tensor, diff: torch.Tensor, expectile: float = 0.7):
    w = torch.where(adv >= 0, expectile, (1 - expectile))
    return w * (diff**2)


class HILPAgent(nn.Module):
    def __init__(
        self,
        config: Namespace,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        load_path: str = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.act_dim = act_dim

        if not isinstance(config, Namespace):
            raise RuntimeError("accept only Namespace as config")
        self.config = copy.deepcopy(config)

        self.networks = self.create_networks(load_path)
        # XXX(Ming): I do not know whether it is required, please have a check
        for k, v in self.networks.items():
            if "target" not in k:
                assert isinstance(v, nn.Module), (k, v)
                self.register_module(k, v)
        self.setup_optimizer()

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters())

    def create_networks(self, load_path: str = None):
        value = GoalConditionedPhiValue(self.obs_dim, self.goal_dim)
        skill_value = GoalConditionedValue(self.obs_dim, self.goal_dim)
        skill_critic = GoalConditionedCritic(self.obs_dim, self.goal_dim, self.act_dim)

        skill_actor = Actor(self.obs_dim, self.goal_dim, self.act_dim)
        return {
            "value": value,
            "target_value": copy.deepcopy(value),
            "skill_value": skill_value,
            "target_skill_value": copy.deepcopy(skill_value),
            "skill_critic": skill_critic,
            "target_skill_critic": copy.deepcopy(skill_critic),
            "skill_actor": skill_actor,
        }

    def get_phi(self, observations: torch.Tensor) -> torch.Tensor:
        """Return embeded observation as goal.

        Args:
            observations (torch.Tensor): Observation tensor.

        Returns:
            torch.Tensor: Embedded observations as goals
        """

        phi = self.networks["value"](observations)
        return phi

    def target_update(self, tau: float = 0.05):
        for k in ["value", "skill_value", "skill_critic"]:
            for tp, p in zip(self.networks[f"target_{k}"], self.networks[k]):
                tp.data.copy_(tau * p + (1 - tau) * tp)

    def forward(
        self,
        method: str,
        observations: torch.Tensor,
        goals: torch.Tensor = None,
        actions: torch.Tensor = None,
    ) -> torch.Tensor:
        if method in ["value", "skill_value", "actor"]:
            return self.networks[method](observations, goals)
        else:
            return self.networks[method](observations, goals, actions)

    @torch.no_grad()
    def compute_target(
        self,
        method: str,
        observations: torch.Tensor,
        goals: torch.Tensor = None,
        actions: torch.Tensor = None,
    ) -> torch.Tensor:
        if method in ["value", "skill_value", "actor"]:
            return self.networks[f"target_{method}"](observations, goals)
        else:
            return self.networks[f"target_{method}"](observations, goals, actions)

    def compute_skill_actor_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        v = self("skill_value", batch["observations"], batch["skills"]).squeeze(-1)
        q = self.compute_target(
            "skill_critic", batch["observations"], batch["skills"], batch["actions"]
        ).squeeze(-1)
        adv = q - v

        exp_a = adv * self.config.skill_temperature
        exp_a = torch.minimum(exp_a, 100.0)

        logits = self("skill_actor", batch["observations"], batch["skills"])
        dist = torch.distributions.categorical.Categorical(logits=logits)
        log_probs = dist.log_prob(batch["actions"]).squeeze(-1)
        actor_loss = -(exp_a * log_probs).mean()
        entropy = dist.entropy().mean()

        return actor_loss, {
            "actor_loss": actor_loss.item(),
            "adv": adv.mean().item(),
            "log_probs": log_probs.mean().item(),
            "entropy": entropy.item(),
        }

    def compute_skill_critic_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # XXX(ming): here is different from the original implementation, which is not use target net
        next_v = self.compute_target(
            "skill_value", batch["next_observations"], batch["skills"]
        ).squeeze(-1)

        target_q = batch["rewards"] + self.config["skill_discount"] * next_v
        q = self(
            "skill_critic", batch["observation"], batch["skills"], batch["actions"]
        ).squeeze(-1)
        critic_loss = ((q - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_max": q.max().item(),
            "q_min": q.min().item(),
            "q_mean": q.mean().item(),
        }

    def compute_skill_value_loss(self, batch):
        q1, q2 = self.network(
            batch["observations"],
            batch["skills"],
            batch["actions"],
            method="skill_target_critic",
        )
        q = np.minimum(q1, q2)
        v = self.network(
            batch["observations"],
            batch["skills"],
            method="skill_value",
            params=network_params,
        )
        adv = q - v
        value_loss = expectile_loss(adv, q - v, self.config.skill_expectile).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v max": v.max().item(),
            "v min": v.min().item(),
            "v mean": v.mean().item(),
            "abs adv mean": torch.abs(adv).mean().item(),
            "adv mean": adv.mean().item(),
            "adv max": adv.max().item(),
            "adv min": adv.min().item(),
            "accept prob": (adv >= 0).float().mean().item(),
        }

    def compute_value_loss(self, batch):
        # masks are 0 if terminal, 1 otherwise
        batch["masks"] = 1.0 - batch["rewards"]
        # rewards are 0 if terminal, -1 otherwise
        batch["rewards"] = batch["rewards"] - 1.0

        next_v = self.compute_target(
            "value", batch["next_observations"], batch["goals"]
        )
        q = batch["rewards"] + self.config.discounte * batch["masks"] * next_v

    def to_torch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        for k, v in batch.items():
            batch[k] = torch.from_numpy(v)
        return batch

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        batch = self.to_torch(batch)
        info = {}
        # goal-conditioned value function
        value_loss, value_info = self.compute_value_loss(batch)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        skill_value_loss, skill_value_info = self.compute_skill_value_loss(batch)
        for k, v in skill_value_info.items():
            info[f"skill_value/{k}"] = v

        skill_critic_loss, skill_critic_info = self.compute_skill_critic_loss(batch)
        for k, v in skill_critic_info.items():
            info[f"skill_critic/{k}"] = v

        skill_actor_loss, skill_actor_info = self.compute_skill_actor_loss(batch)
        for k, v in skill_actor_info.items():
            info[f"skill_actor/{k}"] = v

        loss = value_loss + skill_value_loss + skill_critic_loss + skill_actor_loss

        return loss, info

    @torch.no_grad()
    def sample_skill_actions(
        self,
        observations: torch.Tensor,
        skills: torch.Tensor = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        actions = self.networks["skill_actor"].compute_actions(
            observations, skills, temperature
        )
        actions = torch.clip(actions, -1, 1)
        return actions.cpu().numpy()

    def save(self, f_path: str):
        raise NotImplementedError

    @classmethod
    def load_from(cls, file_path: str):
        import pickle

        with open(file_path, "rb") as f:
            instance = pickle.load(f)
        return instance
