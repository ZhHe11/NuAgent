from typing import Tuple
from collections import namedtuple
from gym import spaces
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from uniagent.models.a2c import ActorCritic
from uniagent.envs.atari import create_atari_env
from uniagent.envs.gym_control import create_gym_control

from application.dqn_gym.policy import AtariPreprocessor


EpisodeState = namedtuple(
    "EpisodeState",
    "obses, dones, actions, net_states, rewards, values, log_probs, entropies, episode_len",
)


def make_env(args):
    if args.task_type == "atari":
        return create_atari_env(args.env_name)
    elif args.task_type == "gym_control":
        return create_gym_control(args.env_name)
    else:
        raise NotImplementedError


class AtariAC(ActorCritic):
    def create_preprocessor(self, num_outputs: int) -> nn.Module:
        return AtariPreprocessor(self.observation_space, num_outputs)


class Agent:
    def __init__(self, args, model_class: nn.Module, model_kwargs: dict):
        self.model = model_class(args=args, **model_kwargs)
        self.device = args.device
        self.env = make_env(args)

    def run_episode(self, args, obs, model, global_counter) -> EpisodeState:
        done = False
