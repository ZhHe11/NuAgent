from typing import Callable, Sequence

import gym
import numpy as np

from .replay_buffer import ReplayBuffer


class Collector:
    def __init__(
        self,
        buffer: ReplayBuffer,
        env: gym.Env,
        action_interface: Callable[[np.ndarray], np.ndarray],
    ):
        self.check_buffer_map(buffer)
        self.buffer = buffer
        self.env = env
        self.action_interface = action_interface

    def check_buffer_map(self, buffer: ReplayBuffer):
        pass

    def collect(self, num_episode: int = 1, seed: int = None):
        for _ in range(num_episode):
            done = False
            observation = self.env.reset(seed)
            while not done:
                action = self.action_interface(observation)
                next_observation, reward, done, info = self.env.step(action)
                self.buffer.push(
                    observation=observation,
                    reward=reward,
                    done=done,
                    next_observation=next_observation,
                )
                observation = next_observation

    def sample(
        self,
        batch_size: int = None,
        indices: Sequence[int] = None,
        to_torch: bool = False,
        device: str = "cpu",
    ):
        return self.buffer.sample(batch_size, indices, to_torch, device)
