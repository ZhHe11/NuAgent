from typing import Sequence

import torch

from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP


def default_init(weights: torch.Tensor):
    nn.init.kaiming_uniform_(weights)


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


class LayerNormMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation=nn.GELU,
        activation_final: bool = False,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim

        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            in_dim = out_dim
            if i + 1 < len(hidden_dims) or activation_final:
                layers.append(activation())
                layers.append(nn.LayerNorm(out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoalConditionedValue(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        norm: bool = True,
    ):
        super().__init__()
        self.net = MLP(
            obs_dim + goal_dim,
            hidden_channels=hidden_dims + [1],
            norm_layer=nn.LayerNorm if norm else None,
        )

    def forward(
        self, observations: torch.Tensor, goals: torch.Tensor = None
    ) -> torch.Tensor:
        if goals is not None:
            return self.net(observations)
        else:
            return self.net(torch.concat([observations, goals], dim=-1))


class GoalConditionedPhiValue:
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        norm: bool = True,
    ):
        super().__init__()
        self.net = MLP(
            obs_dim + goal_dim,
            hidden_channels=hidden_dims + [1],
            norm_layer=nn.LayerNorm if norm else None,
        )

    def forward(
        self, observations: torch.Tensor, goals: torch.Tensor = None
    ) -> torch.Tensor:
        phi_s = self.net(observations).squeeze(-1)
        phi_g = self.net(goals).squeeze(-1)

        squared_dist = ((phi_s - phi_g) ** 2).sum(-1)
        v = -torch.sqrt(torch.maximum(squared_dist, 1e-6))
        return v


class GoalConditionedCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        norm: bool = True,
    ):
        super().__init__()
        self.net = MLP(
            obs_dim + goal_dim + act_dim,
            hidden_channels=hidden_dims + [1],
            norm_layer=nn.LayerNorm if norm else None,
        )

    def forward(
        self,
        observations: torch.Tensor,
        goals: torch.Tensor = None,
        actions: torch.Tensor = None,
    ) -> torch.Tensor:
        if goals is None:
            critic_in = torch.concat([observations, actions], dim=-1)
        else:
            critic_in = torch.concat([observations, goals, actions], dim=-1)
        return self.net(critic_in)


class RNDNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        rep_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        norm: bool = True,
    ):
        super().__init__()
        self.net = MLP(
            obs_dim + goal_dim,
            hidden_channels=hidden_dims + [rep_dim],
            norm_layer=nn.LayerNorm if norm else None,
        )

    def foward(
        self, observations: torch.Tensor, goals: torch.Tensor = None
    ) -> torch.Tensor:
        if goals is not None:
            rets = self.net(observations)
        else:
            rets = self.net(torch.concat([observations, goals], dim=-1))
        rets = (rets**2).sum(-1)
        return rets


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        norm: bool = True,
    ):
        super().__init__()
        self.net = MLP(
            obs_dim + goal_dim,
            hidden_channels=hidden_dims + [act_dim],
            norm_layer=nn.LayerNorm if norm else None,
        )

    def forward(self, observations: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
