from typing import Any, Callable, List

import gym
import numpy as np
import torch

from core.agent import Agent


class GymAgent(Agent):
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        policy_func: Any,
        obs_preprocessor: Callable[[Any, gym.Space], np.array] = None,
        act_preprocessor: Callable[[np.array, gym.Space], Any] = None,
    ):
        super().__init__(
            obs_space, act_space, policy_func, obs_preprocessor, act_preprocessor
        )

    def act(self, observation: List[Any]):
        # convert raw observation to ndarray-like
        obs = np.stack([self.obs_preprocessor(e, self.obs_space) for e in observation])
        # convert observation to torch.Tensor
        obs_tensor = torch.from_numpy(obs)
        act_tensor = self.policy_func(obs_tensor)
        # convert act to environment actions
        actions = act_tensor.cpu().numpy()
        actions = self.act_preprocessor(actions, self.act_space)
        return actions
