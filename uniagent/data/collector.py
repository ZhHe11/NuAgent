from typing import Callable, Sequence

import gym
import numpy as np

from gym.spaces import Discrete

from .replay_buffer import ReplayBuffer


def action_to_onehot(action, n):
    x = np.zeros(n, dtype=np.float32)
    x[action] = 1.0
    return x


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

    def collect(self, batch_size: int = 1, seed: int = None):
        cnt = 0
        while cnt < batch_size:
            done = False
            observation = self.env.reset(seed)
            while not done:
                option, action = self.action_interface(observation)
                next_observation, reward, done, info = self.env.step(action)
                self.buffer.push(
                    observation=observation,
                    action=action_to_onehot(action, self.env.action_space.n)
                    if isinstance(self.env.action_space, Discrete)
                    else action,
                    reward=reward,
                    done=done,
                    next_observation=next_observation,
                    option=option,
                )
                observation = next_observation
                cnt += 1

    def sample(
        self,
        batch_size: int = None,
        indices: Sequence[int] = None,
        to_torch: bool = False,
        device: str = "cpu",
    ):
        return self.buffer.sample(batch_size, indices, to_torch, device)
