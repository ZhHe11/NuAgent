import pytest
import torch

from torch import nn
from torch import optim

from hilp.networks import MLP
from hilp.networks import ModelEnsembling


@pytest.mark.parametrize("num_ensemble", (1, 2, 4))
@pytest.mark.parametrize("out_dim", (1, 2, 4))
def test_model_ensembling_inference(num_ensemble: int, out_dim: int):
    in_channels = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = [
        MLP(
            in_channels=in_channels,
            hidden_channels=(64, 64, out_dim),
            norm_layer=nn.LayerNorm,
        ).to(device)
        for _ in range(num_ensemble)
    ]

    x = torch.randn(2, in_channels, device=device)
    predictions_seperated_models = [model(x) for model in models]
    ensembler = ModelEnsembling(models)

    predictions_ensembler = ensembler(x)
    predictions_stacked = torch.stack(predictions_seperated_models)
    assert predictions_ensembler.shape == predictions_stacked.shape
    assert torch.allclose(
        predictions_ensembler, predictions_stacked, atol=1e-3, rtol=1e-5
    )


def test_model_ensembling_training():
    in_channels = 4
    out_dim = 2
    num_ensemble = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = [
        MLP(
            in_channels=in_channels,
            hidden_channels=(64, 64, out_dim),
            norm_layer=nn.LayerNorm,
        ).to(device)
        for _ in range(num_ensemble)
    ]

    x = torch.randn(2, in_channels, device=device)
    target = torch.rand((2, out_dim), device=device)
    ensembler = ModelEnsembling(models)
    optimizer = optim.Adam(ensembler.parameters(), lr=1e-3)

    while True:
        optimizer.zero_grad()
        pred = ensembler(x)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        for p in ensembler.parameters():
            assert p.grad is not None
        optimizer.step()
        print(loss)
        if loss.cpu().item() < 1e-10:
            break


from typing import Any

from hilp.networks import GoalConditionedValue


@pytest.mark.parametrize("encoder", (None,))
@pytest.mark.parametrize("ensemble_num", (1, 2, 4))
def test_goal_conditioned_with_model_ensemble(ensemble_num: int, encoder: Any):
    obs_dim, goal_dim = 2, 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    goal_value_func = GoalConditionedValue(
        obs_dim, goal_dim, hidden_dims=(64, 64), ensemble_num=ensemble_num, encoder=None
    )
    goal_value_func.to(device)

    obs = torch.rand(2, obs_dim, device=device)
    goal = torch.rand(2, goal_dim, device=device)
    target = torch.rand(2, 1, device=device)

    optimizer = optim.Adam(goal_value_func.parameters(), lr=1e-3)

    while True:
        optimizer.zero_grad()
        pred = goal_value_func(obs, goal)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        for p in goal_value_func.parameters():
            assert p.grad is not None
        for p in goal_value_func.ensembler.parameters():
            assert p.grad is not None
        optimizer.step()
        print(loss)
        if loss.cpu().item() < 1e-10:
            break
