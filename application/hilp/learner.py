from typing import Tuple, Dict, Any

import copy

from functools import partial

import numpy as np
import torch

from torch import nn

from .networks import (
    GoalConditionedPhiValue,
    GoalConditionedValue,
    GoalConditionedCritic,
    Actor,
)


class HILPAgent(nn.Module):
    def __init__(
        self, config: Dict[str, Any], obs_dim: int, goal_dim: int, act_dim: int, load_path: str = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.act_dim = act_dim
        self.config = config

        self.networks = self.create_networks(load_path)
        # XXX(Ming): I do not know whether it is required, please have a check
        for k, v in self.networks.items():
            if 'target' not in k:
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
        phi = self.networks["value"](observations)

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

    @torch.no_grad
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
        
    def compute_skill_actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        v = self('skill_value', batch['observations'], batch['skills']).squeeze(-1)
        q = self.compute_target('skill_critic', batch['observations'], batch['skills'], batch['actions']).squeeze(-1)
        adv = q - v
        
        exp_a = adv * self.config['skill_temperature']
        exp_a = torch.minimum(exp_a, 100.)

        logits = self('skill_actor', batch['observations'], batch['skills'])
        dist = torch.distributions.categorical.Categorical(logits=logits)
        log_probs = dist.log_prob(batch['actions']).squeeze(-1)
        actor_loss = -(exp_a * log_probs).mean()
        entropy = dist.entropy().mean()

        return actor_loss, {
            'actor_loss': actor_loss.item(),
            'adv': adv.mean().item(),
            'log_probs': log_probs.mean().item(),
            'entropy': entropy.item(),
        }
    
    def compute_skill_critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # XXX(ming): here is different from the original implementation, which is not use target net
        next_v = self.compute_target('skill_value', batch['next_observations'], batch['skills']).squeeze(-1)

        target_q = batch['rewards'] + self.config['skill_discount'] * next_v
        q = self('skill_critic', batch['observation'], batch['skills'], batch['actions']).squeeze(-1)
        critic_loss = ((q - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_max': q.max().item(),
            'q_min': q.min().item(),
            'q_mean': q.mean().item()
        }
    
    def compute_skill_value_loss(agent, batch, network_params):
        q1, q2 = agent.network(batch['observations'], batch['skills'], batch['actions'], method='skill_target_critic')
        q = jnp.minimum(q1, q2)
        v = agent.network(batch['observations'], batch['skills'], method='skill_value', params=network_params)
        adv = q - v
        value_loss = expectile_loss(adv, q - v, agent.config['skill_expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v max': v.max(),
            'v min': v.min(),
            'v mean': v.mean(),
            'abs adv mean': jnp.abs(adv).mean(),
            'adv mean': adv.mean(),
            'adv max': adv.max(),
            'adv min': adv.min(),
            'accept prob': (adv >= 0).mean(),
        }
    
    def to_torch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        batch = self.to_torch(batch)


    def save(self, f_path: str):
        raise NotImplementedError

    @classmethod
    def load_from(cls, file_path: str):
        import pickle

        with open(file_path, "rb") as f:
            instance = pickle.load(f)
        return instance
