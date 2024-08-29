import os
# os.environ["MUJOCO_GL"] = "osmesa"
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset

import sys
import os


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
from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
env = FrameStackWrapper(env, 3)

# import gym
# env = gym.make('kitchen-kitchen_relax-v1')

obs = env.reset()


frames = []

# import h5py

# # 打开 HDF5 文件
# with h5py.File('/home/zhangh/.d4rl/datasets/mini_kitchen_microwave_kettle_light_slider-v0.hdf5', 'r') as f:
#     # 列出文件中的所有组
#     print("Keys: %s" % f.keys())
    
#     # 选择一个数据集
#     obs = f['observations'][:]
#     action = f['actions'][:]
#     infos = f['infos'][:]
#     terminals = f['terminals'][:]

# count = 0
# for i in trange(len(action)):
#     if terminals[i] == False:   
#         a = action[i]
#         obs, reward, _, info = env.step(a)
#         # obs_img = info['image']
#         print(env.render(mode='human'))
        
#         # frames.append(obs_img)
        
    
#     else:
#         a = action[i]
#         obs, reward, _, info = env.step(a)
#         obs_img = info['image']
#         frames.append(obs_img)
        
#         gif_name = '/data/zh/project12_Metra/METRA/tests/videos/test_kitchen-relax' + str(count) + '.gif'
#         imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
#         print('saved')
        
#         frames = []
#         count += 1
#         obs = env.reset()

# exit()



load_option_policy_base = torch.load("/data/zh/project12_Metra/METRA/exp/KItchen_her/0709/option_policy9500.pt")
load_traj_encoder_base = torch.load("/data/zh/project12_Metra/METRA/exp/KItchen_her/0709/traj_encoder9500.pt")

policy = load_option_policy_base['policy']
traj_encoder = load_traj_encoder_base['traj_encoder']

obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')


for i in trange(50):
    # policy
    # option = torch.ones(1,load_option_policy_base['dim_option']).to('cuda')
    option = torch.tensor([0,-1], dtype=torch.float).unsqueeze(0).to('cuda')
    obs_option = torch.cat((obs_tensor, option), -1).float()
    action_tensor = policy(obs_option)[1]['mean']
    
    action = action_tensor[0].detach().cpu().numpy()
    obs, reward, _, info = env.step(action)
    obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
    
    for i in info.keys():
        if ' success' in i:
            print(i, info[i])
    
    # for viz
    obs_img = info['image']
    frames.append(obs_img)

gif_name = '/data/zh/project12_Metra/METRA/tests/videos/test_kitchen.gif'
imageio.mimsave(gif_name, frames, 'GIF', duration=1)
print('saved', gif_name)











