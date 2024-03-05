from argparse import Namespace

import torch
import torch.nn as nn

from uniagent.core.agent_runner import EpisodeState


@torch.no_grad()
def compute_gae_and_ret(
    args: Namespace,
    model: nn.Module,
    episode_state: EpisodeState,
    recompute_value: bool = False,
) -> EpisodeState:
    new_episode_state = EpisodeState(episode_state.__dict__)
    R = [0.0] * episode_state.episode_len
    GAE = [0.0] * episode_state.episode_len

    if recompute_value:
        values, _, _ = model(new_episode_state.obses, new_episode_state.net_states)
        next_values, _, _ = model(
            new_episode_state.next_obses, new_episode_state.next_net_states
        )
        new_episode_state.state_values = values
        new_episode_state.next_state_values = next_values

    next_state_values = new_episode_state.next_state_values.cpu().numpy()
    state_values = new_episode_state.state_values.cpu().numpy()
    rewards = new_episode_state.rewards.cpu().numpy()
    ret = 0.0 if new_episode_state.dones[-1] else next_state_values[-1]

    gae = 0.0
    for i in reversed(range(new_episode_state.episode_len)):
        ret = args.gamma * ret + rewards[i]
        delta_t = rewards[i] + args.gamma * next_state_values[i] - state_values[i]
        gae = gae * args.gamma * args.llambda + delta_t

        GAE[i] = gae
        R[i] = ret

    R = torch.FloatTensor(R).to(args.device)
    GAE = torch.FloatTensor(GAE).to(args.device)

    new_episode_state.rets = R
    new_episode_state.gae = GAE

    return new_episode_state
