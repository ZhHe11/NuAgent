import gym

from torch import nn

from uniagent.data.replay_buffer import ReplayBuffer


def create_replay_buffer(env: gym.Env):
    raise NotImplementedError


class DADSAgent(nn.Module):
    pass
