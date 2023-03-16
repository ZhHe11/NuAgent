from typing import Dict, Any, Tuple, Callable, List

import logging
import gym
import numpy as np
import torch

from core.agent_manager import Agent
from core.env_client import run


class Policy:
    def __init__(self):
        pass

    def compute_action(
        self, observation: torch.Tensor, context: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GymAgent(Agent):
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        policy: Any,
        obs_preprocessor: Callable[[Any, gym.Space], np.array] = None,
        act_preprocessor: Callable[[np.array, gym.Space], Any] = None,
    ):
        super().__init__(
            obs_space, act_space, policy, obs_preprocessor, act_preprocessor
        )

    def act(self, observation: List[Any]):
        raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Start agent client.")
    parser.add_argument(
        "--hostname", type=str, required=True, help="env server hostname."
    )
    parser.add_argument(
        "-p", "--port", type=int, required=True, help="env server port."
    )
    parser.add_argument("--env-id", type=str, required=True, help="environment id.")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=4,
        help="episodes will be used for evaluation.",
    )

    args = parser.parse_args()
    agent_policy_mapping = {"default": "default"}
    policies = {"default": Policy()}
    run(
        args.hostname,
        args.port,
        agent_policy_mapping,
        policies,
        args.env_id,
        args.eval_episodes,
    )
