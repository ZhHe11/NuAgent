from typing import Tuple, Any

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
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        for p in self.preprocessor_critic.parameters():
            p.requires_grad = False
        del self.preprocessor_critic

    def create_preprocessor(self, num_outputs: int) -> nn.Module:
        return AtariPreprocessor(self.observation_space, num_outputs)

    def forward(self, obs: Any, state: Tuple[torch.Tensor, torch.Tensor]):
        x, state = self.preprocessor_actor(obs, state)
        # x_critic, _ = self.preprocessor_critic(obs, state)
        return self.critic_linear(x), self.actor_linear(x), state


class AtariLSTMAC(ActorCritic):
    def create_preprocessor(self, num_outputs: int) -> nn.Module:
        raise NotImplementedError
