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

    def collect(
        self, batch_size: int = 1, seed: int = None, max_path_length: int = None
    ):
        cnt = 0
        while cnt < batch_size:
            done = False
            observation = self.env.reset()
            option, action = self.action_interface(observation)
            while not done:
                next_observation, reward, done, info = self.env.step(action)
                next_option, next_action = self.action_interface(next_observation)
                self.buffer.push(
                    observation=np.asarray(observation, dtype=np.float32),
                    action=action_to_onehot(action, self.env.action_space.n)
                    if isinstance(self.env.action_space, Discrete)
                    else action,
                    reward=np.asarray(reward, dtype=np.float32),
                    done=np.asarray(done),
                    next_observation=np.asarray(next_observation, dtype=np.float32),
                    option=np.asarray(option, dtype=np.float32),
                    next_option=np.asarray(next_option, dtype=np.float32),
                )
                observation = next_observation
                action = next_action
                cnt += 1
                done = done or (
                    cnt >= max_path_length if max_path_length is not None else False
                )

    def sample(
        self,
        batch_size: int = None,
        indices: Sequence[int] = None,
        to_torch: bool = False,
        device: str = "cpu",
    ):
        return self.buffer.sample(batch_size, indices, to_torch, device)
