"""
Agent the wrapper of FeuDal, for discrete environment.
"""

from typing import Any, Optional
from threading import Lock
from itertools import count

import argparse
import gym
import torch
import numpy as np
from torch._C import Size

import torch.multiprocessing as mp

from torch.distributions import Categorical, MultivariateNormal
from application.a3c_gym.async_agent import EpisodeState
from application.feudal_gym.async_agent import AsyncAgent as BaseAgent
from uniagent.models.fun.feudal_net import FeudalState
from .net import GPTFeudalVision


class AsyncAgent(BaseAgent):
    def run_episode(
        self,
        obs: np.ndarray,
        done: bool,
        net_state: FeudalState,
        global_counter: mp.Value = None,
        lock: Lock = None,
    ) -> EpisodeState:
        assert isinstance(self.model, GPTFeudalVision)
        self.model.init_memory(1)

        rewards = []
        values_manager = []
        values_worker = []

        log_probs = []
        entropies = []
        obses = []
        actions = []
        dones = []

        if done:
            net_state: FeudalState = self.model.init_state(1, self.device)
        else:
            net_state = self.model.reset_states_grad(net_state)

        counter = count() if not self.model.training else range(self.args.num_steps)

        for _ in counter:
            # net_states_worker.append(
            #     tuple(map(lambda x: x.squeeze(0), net_state.worker_state))
            # )
            # net_states_manager.append(
            #     tuple(map(lambda x: x.squeeze(0), net_state.manager_state))
            # )

            obs = torch.from_numpy(obs).float().to(self.device)
            values, goal_and_actions, log_probs, entropies, net_state = self.model(
                obs.unsqueeze(0), net_state
            )
            manager_values, worker_values = values
