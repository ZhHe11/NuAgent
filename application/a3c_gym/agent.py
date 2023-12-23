from typing import Tuple
from gym import spaces

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from uniagent.models.a2c import ActorCritic
from application.dqn_gym.policy import AtariPreprocessor


class AtariAC(ActorCritic):
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

    def act(
        self, obs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        value, logits, (hx, cx) = self.model(obs, states)

        dist = Categorical(logits=logits)
        if self.model.training:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return value, action, log_prob, entropy, (hx, cx)

    def init_state(self, batch_size: int, device=None):
        states = self.model.init_state(batch_size, device)
        return states
