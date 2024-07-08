import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio

from envs.AntMazeEnv import MazeWrapper
import d4rl

# setup the env
env = MazeWrapper("antmaze-medium-diverse-v0")

env.reset()

frames = []

for i in range(100):
    action =  env.action_space.sample()
    obs, reward, dones, info = env.step(action)
    frames.append(env.render(mode='rgb_array'))

# save the env as gif
path = "./tests/videos/"
path_file = path + "antmaze_test.gif"
imageio.mimsave(path_file, frames, duration=1/24)
print('video saved')


