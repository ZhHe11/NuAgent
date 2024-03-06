from typing import Sequence

import torch

from torch import nn
from torch.nn import functional as F


def default_init(weights: torch.Tensor):
    nn.init.kaiming_uniform_(weights)


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dims: Sequence[int],
        activate_final: bool = False,
        kernel_init=default_init,
    ) -> None:
        super().__init__()

        layers = []
        sizes = [input_size] + hidden_dims + [output_size]
        len_sizes = len(sizes)

        for i in range(len_sizes - 1):
            layer = nn.Linear(sizes[i], sizes[i + 1])
            kernel_init(layer.weight)
            layers.append(layer)
            if i < len_sizes:
                layers.append(nn.ReLU())

        if activate_final:
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return x


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dims: Sequence[int],
        activate_final: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = MLP(input_size, output_size, hidden_dims, activate_final)

    def forward(self, observations: torch.Tensor):
        return self.mlp(observations)


class Critic(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dims: Sequence[int],
        activate_final: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = MLP(input_size, output_size, hidden_dims, activate_final)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        inputs = torch.concat([observations, actions], dim=-1)
        outputs = self.mlp(inputs)
        return outputs.squeeze(-1)


class Policy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, observation: torch.Tensor, temperature: float = 1.0):
        pass
