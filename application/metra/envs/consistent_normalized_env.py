import gym
import akro

from dataclasses import dataclass


@dataclass(frozen=True)
class InOutSpec:
    """Describes the input and output spaces of a primitive or module."""

    input_space: akro.Space
    output_space: akro.Space


@dataclass(frozen=True, init=False)
class EnvSpec(InOutSpec):
    """Describes the observations, actions, and time horizon of an MDP.

    Args:
        observation_space (akro.Space): The observation space of the env.
        action_space (akro.Space): The action space of the env.
        max_episode_length (int): The maximum number of steps allowed in an
            episode.

    """

    def __init__(self, observation_space, action_space, max_episode_length=None):
        object.__setattr__(self, "max_episode_length", max_episode_length)
        super().__init__(input_space=action_space, output_space=observation_space)

    max_episode_length: int = None

    @property
    def action_space(self):
        """Get action space.

        Returns:
            akro.Space: Action space of the env.

        """
        return self.input_space

    @property
    def observation_space(self):
        """Get observation space of the env.

        Returns:
            akro.Space: Observation space.

        """
        return self.output_space

    @action_space.setter
    def action_space(self, action_space):
        """Set action space of the env.

        Args:
            action_space (akro.Space): Action space.

        """
        self._input_space = action_space

    @observation_space.setter
    def observation_space(self, observation_space):
        """Set observation space of the env.

        Args:
            observation_space (akro.Space): Observation space.

        """
        self._output_space = observation_space


class AkroWrapperTrait:
    @property
    def spec(self):
        return EnvSpec(
            action_space=akro.from_gym(self.action_space),
            observation_space=akro.from_gym(self.observation_space),
        )


import numpy as np


class ConsistentNormalizedEnv(AkroWrapperTrait, gym.Wrapper):
    def __init__(
        self,
        env,
        expected_action_scale=1.0,
        flatten_obs=True,
        normalize_obs=True,
        mean=None,
        std=None,
    ):
        super().__init__(env)

        self._normalize_obs = normalize_obs
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        self._obs_mean = np.full(
            env.observation_space.shape, 0 if mean is None else mean
        )
        self._obs_var = np.full(
            env.observation_space.shape, 1 if std is None else std**2
        )

        self._cur_obs = None

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_space = akro.Box(
                low=-self._expected_action_scale,
                high=self._expected_action_scale,
                shape=self.env.action_space.shape,
            )
        else:
            self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _apply_normalize_obs(self, obs):
        normalized_obs = (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
        return normalized_obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._cur_obs = obs

        if self._normalize_obs:
            obs = self._apply_normalize_obs(obs)

        if self._flatten_obs:
            obs = gym.spaces.utils.flatten(self.env.observation_space, obs)

        return obs

    def step(self, action, **kwargs):
        if isinstance(self.env.action_space, gym.spaces.Box):
            # rescale the action when the bounds are not inf
            lb, ub = self.env.action_space.low, self.env.action_space.high
            if np.all(lb != -np.inf) and np.all(ub != -np.inf):
                scaled_action = lb + (action + self._expected_action_scale) * (
                    0.5 * (ub - lb) / self._expected_action_scale
                )
                scaled_action = np.clip(scaled_action, lb, ub)
            else:
                scaled_action = action
        else:
            scaled_action = action

        next_obs, reward, done, info = self.env.step(scaled_action, **kwargs)
        info["original_observations"] = self._cur_obs
        info["original_next_observations"] = next_obs

        self._cur_obs = next_obs

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)

        if self._flatten_obs:
            next_obs = gym.spaces.utils.flatten(self.env.observation_space, next_obs)

        return next_obs, reward, done, info


consistent_normalize = ConsistentNormalizedEnv
