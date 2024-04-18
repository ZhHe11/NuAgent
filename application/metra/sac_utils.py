from typing import Dict

import gym
import torch

from torch import nn
from torch.nn import functional as F


def _clip_actions(
    algo: nn.Module, action_space: gym.Space, actions: torch.Tensor
) -> torch.Tensor:
    epsilon = 1e-6
    lower = torch.from_numpy(action_space.low).to(algo.config.device) + epsilon
    upper = torch.from_numpy(action_space.high).to(algo.config.device) - epsilon

    clip_up = (actions > upper).float()
    clip_down = (actions < lower).float()
    with torch.no_grad():
        clip = (upper - actions) * clip_up + (lower - actions) * clip_down

    return actions + clip


def compute_loss_qf(
    algo: nn.Module,
    action_space: gym.Space,
    batch: Dict[str, torch.Tensor],
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    dones: torch.Tensor,
    rewards: torch.Tensor,
    policy: nn.Module,
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    q_ensemble: torch.Tensor = algo.qf(obs, actions).squeeze(-1)

    next_action_dists, *_ = policy(next_obs)
    if hasattr(next_action_dists, "rsample_with_pre_tanh_value"):
        (
            new_next_actions_pre_tanh,
            new_next_actions,
        ) = next_action_dists.rsample_with_pre_tanh_value()
        new_next_action_log_probs = next_action_dists.log_prob(
            new_next_actions, pre_tanh_value=new_next_actions_pre_tanh
        )
    else:
        new_next_actions = next_action_dists.rsample()
        new_next_actions = _clip_actions(algo, action_space, new_next_actions)
        new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)

    target_q_values: torch.Tensor = (
        algo.target_qf(next_obs, new_next_actions).squeeze(-1).min(dim=0)[0]
    )

    assert target_q_values.shape == new_next_action_log_probs.shape, (
        target_q_values.shape,
        new_next_action_log_probs.shape,
    )
    target_q_values = target_q_values - alpha * new_next_action_log_probs
    target_q_values = target_q_values * algo.config.discount

    with torch.no_grad():
        assert rewards.shape == target_q_values.shape == dones.shape, (
            rewards.shape,
            target_q_values.shape,
            dones.shape,
        )
        q_target = rewards + target_q_values * (1.0 - dones)
        q_target_ensemble = torch.ones(
            q_ensemble.size(0), 1, requires_grad=False
        ).matmul(q_target.unsqueeze(0))

    assert q_ensemble.shape == q_target_ensemble.shape, (
        q_ensemble.shape,
        q_target_ensemble.shape,
    )
    loss = F.mse_loss(q_ensemble.flatten(), q_target_ensemble.flatten()) * 0.5

    return loss, {
        "QTargetsMean": q_target.mean().cpu().item(),
        "QTdErrsMean": (q_target_ensemble.flatten() - q_ensemble.flatten()).mean(),
        "LossQf": loss.cpu().item(),
    }


def compute_loss_sacp(
    algo: nn.Module,
    batch: Dict[str, torch.Tensor],
    obs: torch.Tensor,
    policy: nn.Module,
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    action_dists, *_ = policy(obs)
    if hasattr(action_dists, "rsample_with_pre_tanh_value"):
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        new_action_log_probs = action_dists.log_prob(
            new_actions, pre_tanh_value=new_actions_pre_tanh
        )
    else:
        new_actions = action_dists.rsample()
        new_actions = _clip_actions(algo, new_actions)
        new_action_log_probs = action_dists.log_prob(new_actions)

    q_values_ensemble: torch.Tensor = algo.qf(obs, new_actions)
    min_q_values = q_values_ensemble.min(dim=0)[0]

    loss_sacp = (alpha * new_action_log_probs - min_q_values).mean()

    batch.update(
        {
            "new_action_log_probs": new_action_log_probs,
        }
    )

    return loss_sacp, {
        "SacpNewActionLogProbMean": new_action_log_probs.mean(),
        "LossSacp": loss_sacp.cpu().item(),
    }


def compute_loss_alpha(
    algo: nn.Module,
    batch: Dict[str, torch.Tensor],
):
    loss_alpha = (
        -algo.log_alpha.param
        * (batch["new_action_log_probs"].detach() + algo._target_entropy)
    ).mean()

    return loss_alpha, {
        "Alpha": algo.log_alpha.param.exp(),
        "LossAlpha": loss_alpha.cpu().item(),
    }
