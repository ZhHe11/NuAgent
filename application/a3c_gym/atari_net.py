from typing import Tuple

from gym import Space

import torch
import torch.nn as nn

from uniagent.models.a2c import ActorCritic
from application.dqn_gym.policy import AtariPreprocessor


class AtariLSTMPreprocessor(AtariPreprocessor):
    def __init__(self, observation_space: Space, num_outputs: int) -> None:
        super().__init__(observation_space, num_outputs)
        self.lstm = nn.LSTMCell(num_outputs, 256)

    def forward(
        self, obs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embed(obs)
        x, states = self.lstm(x, states)
        return x, states

    def init_state(self, batch_size: int, device: torch.DeviceObjType = None):
        if device is None:
            device = next(self.parameters()).device
        hx = torch.zeros(batch_size, 256).to(device)
        cx = torch.zeros(batch_size, 256).to(device)
        return (hx, cx)


class AtariAC(ActorCritic):
    def create_preprocessor(self, num_outputs: int) -> nn.Module:
        return AtariPreprocessor(self.observation_space, num_outputs)


class AtariLSTMAc(ActorCritic):
    def create_preprocessor(self, num_outputs: int) -> nn.Module:
        raise NotImplementedError
