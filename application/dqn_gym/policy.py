from typing import Tuple
from gym import spaces

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniagent.models.dqn import DQN
from uniagent.models.torch_net_utils import View


class AtariPreprocessor(nn.Module):
    def __init__(self, observation_space: spaces.Space, num_outputs: int) -> None:
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(3136, num_outputs),
            nn.ReLU(),
        )

    def forward(
        self, obs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embed(obs)
        return x, states

    def init_state(self, batch_size: int, device: torch.DeviceObjType = None):
        if device is None:
            device = next(self.parameters()).device
        hx = torch.zeros(batch_size, 2).to(device)
        cx = torch.zeros(batch_size, 2).to(device)
        return (hx, cx)


class AtariDQN(DQN):
    def create_preprocessor(self, num_outputs: int) -> nn.Module:
        return AtariPreprocessor(self.observation_space, num_outputs)


class Agent:
    def __init__(
        self,
        model_class: nn.Module,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.DeviceObjType = None,
    ) -> None:
        self.device = device
        self.action_space = action_space
        self.model = model_class(observation_space, action_space).to(device)
        self.target_model = copy.deepcopy(self.model)

        for p in self.target_model.parameters():
            p.requires_grad = False

        # linear schedule for epsilon
        self.eps = 1.0
        self.eps_decay = 1e-5
        self.eps_min = 0.01

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        q, states = self.model(obs, states)
        if self.model.training:
            # then epsilon-greedy
            if random.random() < self.eps:
                batch_size = obs.size(0)
                action = torch.randint(self.action_space.n, (batch_size,))
            else:
                action = q.argmax(dim=-1)
            self.eps = max(self.eps - self.eps_decay, self.eps_min)
        else:
            action = q.argmax(dim=-1)

        return action, states

    def compute_q(
        self, obs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.model(obs, states)

    @torch.no_grad()
    def compute_target_q(
        self, obs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.target_model(obs, states)

    def soft_update(self, tau: float = 0.05):
        for target_param, param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def init_state(self, batch_size: int, device=None):
        states = self.model.init_state(batch_size, device)
        return states
