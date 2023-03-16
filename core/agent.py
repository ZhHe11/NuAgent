from typing import Any, Callable, List

import gym
import numpy as np


def default_obs_preprocessor(obs: Any, obs_space: gym.Space):
    return obs


def default_act_preprocessor(act: Any, act_space: gym.Space):
    return act


class Agent:
    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        policy: Any,
        obs_preprocessor: Callable[[Any, gym.Space], np.array] = None,
        act_preprocessor: Callable[[np.array, gym.Space], Any] = None,
    ):
        self.policy = policy
        self.obs_space = obs_space
        self.act_space = act_space
        self.obs_preprocessor = obs_preprocessor or default_obs_preprocessor
        self.act_preprocessor = act_preprocessor or default_act_preprocessor

    def act(self, observation: List[Any]):
        """Convert raw environment observation as tensor array like.

        Args:
            observation (List[Any]): A list of raw environment observation.
        """

        if self.policy is None:
            return [self.act_space.sample() for obs in observation]
        else:
            raise NotImplementedError
