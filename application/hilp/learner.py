from typing import Tuple, Dict, Any, Sequence, Union

import copy
import logging

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


logger = logging.getLogger("HILPAgent")


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
        self.setup_optimizer()

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.config.lr)

    def create_networks(self, load_path: str = None) -> nn.ModuleDict:
        """Register a module dict.

        Args:
            load_path (str, optional): File path of local storage. Defaults to None.

        Returns:
            nn.ModuleDict: An instance of module dict.
        """

        value = GoalConditionedPhiValue(
            self.obs_dim, self.goal_dim, self.config.skill_dim
        ).to(self.config.device)
        skill_value = GoalConditionedValue(self.obs_dim, self.config.skill_dim).to(
            self.config.device
        )
        skill_critic = GoalConditionedCritic(
            self.obs_dim, self.config.skill_dim, self.act_dim
        ).to(self.config.device)
        skill_actor = Actor(self.obs_dim, self.config.skill_dim, self.act_dim).to(
            self.config.device
        )

        return nn.ModuleDict(
            {
                "value": value,
                "target_value": copy.deepcopy(value),
                "skill_value": skill_value,
                "target_skill_value": copy.deepcopy(skill_value),
                "skill_critic": skill_critic,
                "target_skill_critic": copy.deepcopy(skill_critic),
                "skill_actor": skill_actor,
            }
        )

    def get_phi(self, observations: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Return embeded observation as goal.

        Args:
            observations (torch.Tensor): Observation tensor.

        Returns:
            torch.Tensor: Embedded observations as goals
        """

        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).to(
                dtype=torch.float32, device=self.config.device
            )
        phi = self.networks["value"].get_phi(observations)
        return phi

    def target_update(self):
        tau = self.config.tau
        for k in ["value", "skill_value", "skill_critic"]:
            for tp, p in zip(
                self.networks[f"target_{k}"].parameters(), self.networks[k].parameters()
            ):
                tp.data.copy_(tau * p + (1 - tau) * tp)

    def forward(
        self,
        method: str,
        observations: torch.Tensor,
        goals: torch.Tensor = None,
        actions: torch.Tensor = None,
    ) -> torch.Tensor:
        if method in ["value", "skill_value", "skill_actor"]:
            return self.networks[method](observations, goals)
        elif method in ["skill_critic"]:
            return self.networks[method](observations, goals, actions)
        else:
            raise ValueError(f"unexpected method: {method}")

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
        with torch.no_grad():
            v = self("skill_value", batch["observations"], batch["skills"]).squeeze(-1)
        q = self.compute_target(
            "skill_critic", batch["observations"], batch["skills"], batch["actions"]
        ).squeeze(-1)
        adv = q - v

        exp_a = adv * self.config.skill_temperature
        exp_a = torch.clamp(exp_a, max=100.0)

        dist: torch.distributions.Distribution = self(
            "skill_actor", batch["observations"], batch["skills"]
        )
        log_probs = dist.log_prob(batch["actions"]).squeeze(-1)
        assert exp_a.size() == log_probs.size(), (exp_a.size(), log_probs.size())
        # actor_loss = -(exp_a * log_probs).mean()
        actor_loss = -log_probs.mean()
        entropy = dist.entropy().mean()

        info = {
            "actor_loss": actor_loss.item(),
            "adv": adv.mean().item(),
            "bc_log_probs": log_probs.mean().item(),
            "adv_median": adv.median().item(),
            "mse": ((dist.mode - batch["actions"]) ** 2).mean().item(),
            "entropy": entropy.item(),
        }

        return actor_loss, info

    def compute_skill_critic_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # XXX(ming): here is different from the original implementation, which is not use target net
        next_v = self.compute_target(
            "skill_value", batch["next_observations"], batch["skills"]
        ).squeeze(-1)

        target_q = batch["rewards"] + self.config.skill_discount * next_v
        q = self(
            "skill_critic", batch["observations"], batch["skills"], batch["actions"]
        ).squeeze(-1)
        critic_loss = ((q - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q max": q.max().item(),
            "q min": q.min().item(),
            "q mean": q.mean().item(),
        }

    def compute_skill_value_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        q = self.compute_target(
            "skill_critic", batch["observations"], batch["skills"], batch["actions"]
        ).squeeze()
        v = self("skill_value", batch["observations"], batch["skills"]).squeeze()
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

    def compute_value_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # masks are 0 if terminal, 1 otherwise
        batch["masks"] = 1.0 - batch["rewards"]
        # rewards are 0 if terminal, -1 otherwise
        batch["rewards"] = batch["rewards"] - 1.0

        next_v_target = self.compute_target(
            "value", batch["next_observations"], batch["goals"]
        ).squeeze()
        q_target = (
            batch["rewards"] + self.config.discount * batch["masks"] * next_v_target
        )
        v_target = self.compute_target(
            "value", batch["observations"], batch["goals"]
        ).squeeze()
        adv = q_target - v_target

        v = self("value", batch["observations"], batch["goals"])
        value_loss = expectile_loss(adv, q_target - v, self.config.expectile).mean()

        return value_loss, {
            "value_loss": value_loss.item(),
            "v max": v.max().item(),
            "v min": v.min().item(),
            "v mean": v.mean().item(),
            "abs adv mean": adv.abs().mean().item(),
            "adv mean": adv.mean().item(),
            "adv min": adv.min().item(),
            "adv max": adv.max().item(),
            "accept prob": (adv >= 0).float().mean().item(),
        }

    def to_torch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).to(
                    dtype=torch.float32, device=self.config.device
                )
        return batch

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        batch = self.to_torch(batch)
        info = {}
        # goal-conditioned value function
        value_loss, value_info = self.compute_value_loss(batch)
        for k, v in value_info.items():
            info[f"value/{k}"] = v
        logger.debug(f"value_info: {value_info}")

        # for skill policy
        bsize = batch["observations"].shape[0]
        with torch.no_grad():
            batch["phis"] = self.get_phi(batch["observations"])
            batch["next_phis"] = self.get_phi(batch["next_observations"])
        random_skills = (
            torch.randn(bsize, self.config.skill_dim)
            .float()
            .to(device=self.config.device)
        )
        batch["skills"] = random_skills / torch.linalg.norm(
            random_skills, dim=1, keepdim=True
        )
        batch["rewards"] = ((batch["next_phis"] - batch["phis"]) * batch["skills"]).sum(
            axis=1
        )
        batch = self.to_torch(batch)

        skill_value_loss, skill_value_info = self.compute_skill_value_loss(batch)
        for k, v in skill_value_info.items():
            info[f"skill_value/{k}"] = v
        logger.debug(f"skill_value_info: {skill_value_info}")

        skill_critic_loss, skill_critic_info = self.compute_skill_critic_loss(batch)
        for k, v in skill_critic_info.items():
            info[f"skill_critic/{k}"] = v
        logger.debug(f"skill_critic_info: {skill_critic_info}")

        skill_actor_loss, skill_actor_info = self.compute_skill_actor_loss(batch)
        for k, v in skill_actor_info.items():
            info[f"skill_actor/{k}"] = v
        logger.debug(f"skill_actor_info: {skill_actor_info}")

        loss = value_loss + skill_value_loss + skill_critic_loss + skill_actor_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_update()

        return info

    @torch.no_grad()
    def sample_skill_actions(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        skills: Union[np.ndarray, torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).to(
                dtype=torch.float32, device=self.config.device
            )
        if skills is not None and isinstance(skills, np.ndarray):
            skills = torch.from_numpy(skills).to(
                dtype=torch.float32, device=self.config.device
            )
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
