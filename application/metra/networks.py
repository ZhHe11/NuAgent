from typing import Sequence

import torch

from torch import nn
from torch import vmap
from torch.func import stack_module_state, functional_call, grad

from torchvision.ops import MLP

from application.hilp.networks import ModelEnsembling, Actor


class ParameterModule(nn.Module):
    def __init__(self, init_value):
        super().__init__()

        self.param = torch.nn.Parameter(init_value)


class ContinuousMLPQFunctionEx(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        norm: bool = True,
        ensemble_num: int = 2,
        encoder: nn.Module = None,
    ) -> None:
        super().__init__()

        _nets = [
            MLP(
                obs_dim + action_dim,
                hidden_channels=hidden_dims + (1,),
                norm_layer=nn.LayerNorm if norm else None,
            )
            for _ in range(ensemble_num)
        ]
        if ensemble_num > 1:
            net = ModelEnsembling(_nets)
        else:
            net = _nets[0]
        if encoder is not None:
            self.net = nn.Sequential([encoder(), net])
        else:
            self.net = net

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        inputs = torch.concat([observations, actions], dim=-1)
        outputs = self.net(inputs)
        return outputs.squeeze(-1)


class GaussianMLP(Actor):
    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = self.net(x)
        mean = self.mean_layer(x)
        log_std = torch.clip(self.log_std_layer(x), self.log_std_min, self.log_std_max)
        assert mean.shape == log_std.shape, (mean.shape, log_std.shape)
        dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=torch.exp(log_std)),
            1,
        )
        if self.tanh_squash_distribution:
            raise NotImplementedError
        return dist


from typing import Any

import numpy as np


class Policy(Actor):
    def process_observations(self, observation: Any):
        if isinstance(observation, torch.Tensor):
            return observation
        elif isinstance(observation, np.ndarray):
            return torch.from_numpy(observation).float()
        else:
            raise RuntimeError

    def forward(
        self, observation_with_goals: torch.Tensor, temperature: float = 1.0
    ) -> torch.distributions.Distribution:
        x = self.net(observation_with_goals)
        mean = self.mean_layer(x)
        log_std = torch.clip(self.log_std_layer(x), self.log_std_min, self.log_std_max)
        assert mean.shape == log_std.shape, (mean.shape, log_std.shape)
        dist = torch.distributions.Independent(
            torch.distributions.Normal(
                loc=mean, scale=torch.exp(log_std) * temperature
            ),
            1,
        )
        if self.tanh_squash_distribution:
            raise NotImplementedError
        return dist
