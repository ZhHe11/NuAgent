import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio

import lexa.d4rl as d4rl
import torch
from sklearn.decomposition import PCA
import matplotlib.cm as cm

import os
import sys
local_lexa_path = os.path.abspath('./lexa')
sys.path.insert(0, local_lexa_path)
from envs.lexa.mykitchen import MyKitchenEnv

import imageio
from tqdm import trange, tqdm

env = MyKitchenEnv(log_per_goal=True)
obs = env.reset()
frames = []

# import pdb; pdb.set_trace()

for i in trange(50):
    action = env.action_space.sample()
    obs, reward, _, info = env.step(action)
    obs_img = obs.reshape(64,64,3)
    frames.append(obs_img)
    print(obs)
    
gif_name = 'test_kitchen.gif'
imageio.mimsave(gif_name, frames, 'GIF', duration=1)

    









