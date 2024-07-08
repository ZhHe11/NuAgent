import akro
import gym
import numpy as np
from envs.maze_env import MazeEnv
import matplotlib.pyplot as plt

import d4rl

# env = MazeEnv(
#                 max_path_length=100, 
#                 action_range=0.2,
#             )


env = gym.make('AntMaze')

env.reset()

traj = []

for i in range(100):
    action =  env.action_space.sample()
    obs, reward, dones, info = env.step(action)
    traj.append(obs)

traj_len = len(traj)
color = ['red']

# fig, ax = plt.subplots()

# env.render_trajectories(trajectories=traj, colors=color, plot_axis=None, ax=ax)

# env.plot_trajectories(trajectories=traj, colors=color, plot_axis=None, ax=ax)

