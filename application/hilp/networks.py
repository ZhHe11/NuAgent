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


# see the use of model ensembling: https://pytorch.org/tutorials/intermediate/ensembling.html
from typing import List

import copy

from functools import partial
from torch import vmap
from torch.func import stack_module_state, functional_call, grad


class ModelEnsembling(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()

        # Construct a "stateless" version of one of the models.
        # It is "stateless" in the sense that the parameters are
        #   meta Tensors and do not have storage.
        base_net = copy.deepcopy(models[0]).to("meta")
        self.vmap_params, self.vmap_buffers = stack_module_state(models)
        self.vmap_params = {
            k: nn.Parameter(v, requires_grad=True) for k, v in self.vmap_params.items()
        }

        # wrap tensor as parameter
        for k, v in self.vmap_params.items():
            self.register_parameter(k.replace(".", "_"), v)

        for k, v in self.vmap_buffers.items():
            self.register_buffer(k.replace(".", "_"), v)

        def fmodel(vmap_params, vmap_buffers, args, kwargs):
            return functional_call(base_net, (vmap_params, vmap_buffers), args, kwargs)

        self.fmodel = fmodel

    def forward(self, *args, **kwargs):
        rets_vmap = vmap(self.fmodel, in_dims=(0, 0, None, None))(
            self.vmap_params, self.vmap_buffers, args, kwargs
        )
        return rets_vmap


class GoalConditionedValue(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        norm: bool = True,
        ensemble_num: int = 2,
        encoder: nn.Module = None,
    ):
        super().__init__()
        _nets = [
            MLP(
                obs_dim + goal_dim,
                hidden_channels=hidden_dims + (1,),
                norm_layer=nn.LayerNorm if norm else None,
            )
            for _ in range(ensemble_num)
        ]
        net = ModelEnsembling(_nets)
        if encoder is not None:
            self.net = nn.Sequential([encoder(), net])
        else:
            self.net = net

    def forward(
        self, observations: torch.Tensor, goals: torch.Tensor = None
    ) -> torch.Tensor:
        if goals is None:
            return self.net(observations)
        else:
            return self.net(torch.concat([observations, goals], dim=-1))


class GoalConditionedPhiValue(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        skill_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        norm: bool = True,
        ensemble_num: int = 2,
        encoder: nn.Module = None,
    ):
        super().__init__()
        assert obs_dim == goal_dim, (goal_dim, obs_dim)
        _nets = [
            MLP(
                obs_dim,
                hidden_channels=hidden_dims + (skill_dim,),
                norm_layer=nn.LayerNorm if norm else None,
            )
            for _ in range(ensemble_num)
        ]
        net = ModelEnsembling(_nets)
        if encoder is not None:
            self.net = nn.Sequential([encoder(), net])
        else:
            self.net = net

    def get_phi(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, observations: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        phi_s = self.net(observations)
        phi_g = self.net(goals)

        squared_dist = ((phi_s - phi_g) ** 2).sum(-1)
        v = -torch.sqrt(torch.clamp(squared_dist, min=1e-6))
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
            hidden_channels=hidden_dims + (1,),
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


import numpy as np


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        norm: bool = True,
        state_dependent_std: bool = True,
    ):
        super().__init__()
        self.net = MLP(
            obs_dim + goal_dim,
            hidden_channels=hidden_dims,
            norm_layer=nn.LayerNorm if norm else None,
        )
        mean = nn.Linear(hidden_dims[-1], act_dim, bias=True)
        if state_dependent_std:
            log_stds = nn.Linear(hidden_dims[-1], act_dim, bias=True)
            nn.init.xavier_uniform_(log_stds.weight)
        else:
            log_stds = torch.autograd.Variable(torch.rand(act_dim))
        # log_stds = torch.clip(log_stds, log_std_min, log_std_max)
        self.mean_layer = mean
        self.log_std_layer = log_stds

    def compute_actions(
        self, observations: torch.Tensor, goals: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        dist = self(observations, goals, temperature)
        actions = dist.sample()
        return actions

    def forward(
        self, observations: torch.Tensor, goals: torch.Tensor, temperature: float = 1.0
    ) -> torch.distributions.Distribution:
        x = self.net(torch.concat([observations, goals], dim=-1))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # import pdb; pdb.set_trace()
        # dist = torch.distributions.MultivariateNormal(
        #     loc=mean, scale_tril=torch.exp(log_std) * temperature
        # )
        dist = torch.distributions.Independent(
            torch.distributions.Normal(
                loc=mean, scale=torch.exp(log_std) * temperature
            ),
            1,
        )
        return dist
