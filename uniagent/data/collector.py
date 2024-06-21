from typing import Callable, Sequence

import gym
import numpy as np

from gym.spaces import Discrete

from .replay_buffer import ReplayBuffer

from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing as mp
import copy

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
        self.empty_buffer = copy.deepcopy(buffer) 

    def check_buffer_map(self, buffer: ReplayBuffer):
        pass

    def collect(
        self, batch_size: int = 1, seed: int = None, max_path_length: int = None
    ):
        
        buffer = self.empty_buffer

        cnt = 0
        while cnt < batch_size:
            done = False
            observation = self.env.reset()
            # 这里有问题，metra的方法应该是，每条traj使用同一个option
            # 第一次采样option的时候，是随机的，但是现在使用的是固定的option，就是[1,0][0,1]这种；
            option, action = self.action_interface.get_option_action(observation)
            while not done:
                next_observation, reward, done, info = self.env.step(action)
                next_action = self.action_interface.get_action(next_observation, option)
                next_option = option
                buffer.push(
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


