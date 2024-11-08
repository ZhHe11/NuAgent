import gym
import d4rl

from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
import torch

from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze, plot_trajectories, plot_value, trajectory_image

import matplotlib.cm as cm
from tqdm import trange, tqdm

env = MazeWrapper("maze2d-umaze-v1", random_init=False)
obs = env.reset()
max_path_length=300

paths = "/mnt/nfs2/zhanghe/NuAgent/.option_policy-500.pt"

policy_path = paths
traj_encoder_path = policy_path.replace('option_policy', 'traj_encoder')

load_option_policy_base = torch.load(policy_path)
load_traj_encoder_base = torch.load(traj_encoder_path)
agent_policy = load_option_policy_base['policy'].eval()
if "target_traj_encoder" in load_traj_encoder_base.keys():
    agent_traj_encoder = load_traj_encoder_base['target_traj_encoder'].eval()
else:
    agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()

env.reset()
frames = []
num_eval = 3
dim_option = 2
device = 'cuda'

fig, axes = plt.subplots(1,2, figsize=(20,10))
ax1 = axes[0]
ax2 = axes[1]
np_random = np.random.default_rng(seed=0) 
goal_list = []
init_obs = env.reset()  
random_options = np.random.uniform(-1,1, (num_eval, dim_option))


All_trajs_list = []
All_Repr_obs_list = []

for i in range(num_eval):
    Repr_obs_list = []
    Repr_goal_list = []
    Obs_list = []
    obs = env.reset()
    obs = torch.tensor(obs).unsqueeze(0).to(device).float()
    option = torch.tensor(random_options[i]).unsqueeze(0).to(device)
    Repr_goal_list.append(option.cpu().numpy()[0])
    traj_list = {}
    traj_list["observation"] = []
    traj_list["info"] = []

    for i in trange(max_path_length):
        phi_obs_ = agent_traj_encoder(obs).mean
        obs_option = torch.cat((obs, option), -1).float()

        action, agent_info = agent_policy.get_action(obs_option)
        obs, reward, dones, info = env.step(action)

        traj_list["observation"].append(obs)
        info['x'], info['y'] = obs[0], obs[1]
        traj_list["info"].append(info)
        Repr_obs_list.append(phi_obs_.detach().cpu().numpy()[0])    
        Obs_list.append(obs[:2])

        obs = torch.tensor(obs).unsqueeze(0).to(device).float()

    All_trajs_list.append(traj_list)
    All_Repr_obs_list.append(Repr_obs_list)


Obs_array = np.array(Obs_list)
Repr_obs_array = np.array(All_Repr_obs_list)
# Repr_goal_array = np.array(Repr_goal_list)
# ax1.scatter(Obs_array[:,0], Obs_array[:,1], c='r')

plot_trajectories(env, All_trajs_list, fig, ax1)
colors = cm.rainbow(np.linspace(0, 1, len(All_Repr_obs_list)))
for i in range(Repr_obs_array.shape[0]):
    ax2.scatter(Repr_obs_array[i,:,0], Repr_obs_array[i,:,1], c=colors[i])
    # ax2.scatter(Repr_goal_array[:,0], Repr_goal_array[:,1], marker='*', c=colors)
plt.savefig('lm.png')
print('saved at ', './lm.png')








