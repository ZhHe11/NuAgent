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
import imageio

import copy

## def functions
def Psi(phi_x, phi_x0=None, max_path_length=300):
    return torch.tanh(2/max_path_length * (phi_x))

env = MazeWrapper("maze2d-large-v1", random_init=False)
obs = env.reset()
max_path_length=300

paths = "/mnt/nfs2/zhanghe/NuAgent/exp/LittleMaze/SZNsd000_1731046347_lm_PSZP/wandb/run-20241108_141229-njkzb10t/filesoption_policy-1100.pt"
filepath = './'
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
num_eval = 10
dim_option = 2
device = 'cuda'
# eval_type = 'random_psi'
eval_type = 'goal_psi'
save_gif = False

fig, axes = plt.subplots(2,2, figsize=(8,6))
ax1 = axes[0,0]
ax2 = axes[1,0]
np_random = np.random.default_rng(seed=0) 
GoalList = np.load('/mnt/nfs2/zhanghe/NuAgent/tests/savenp/less-LargeMazeGoal.npy')


init_obs = env.reset() 
random_options = np.random.uniform(-1,1, (num_eval, dim_option))

All_trajs_list = []
All_Repr_obs_list = []
All_Repr_goal_list = []

if eval_type == 'goal_psi':
    num_eval = len(GoalList)

for i in trange(num_eval):
    Repr_obs_list = []
    Obs_list = []
    obs = env.reset()
    obs = torch.tensor(obs).unsqueeze(0).to(device).float()

    if eval_type in ['random', 'random_psi'] :
        option = torch.tensor(random_options[i]).unsqueeze(0).to(device)
    elif eval_type == 'goal_psi':
        goal = GoalList[i]
        tensor_goal = torch.tensor(goal).to(device)
        obs_goal = copy.deepcopy(obs)
        obs_goal = env.get_target_obs(obs_goal, tensor_goal)
        Phi_g = agent_traj_encoder(obs_goal).mean
        option = Psi(Phi_g)

    All_Repr_goal_list.append(option.detach().cpu().numpy()[0])
    traj_list = {}
    traj_list["observation"] = []
    traj_list["info"] = []
    frames = []

    for t in trange(max_path_length):
        # if eval_type == 'goal_psi':
        #     obs_goal = copy.deepcopy(obs)
        #     obs_goal = env.get_target_obs(obs_goal, tensor_goal)
        #     Phi_g = agent_traj_encoder(obs_goal).mean
        #     option = Psi(Phi_g)
        phi_obs_ = agent_traj_encoder(obs).mean
        obs_option = torch.cat((obs,option ), -1).float()

        action, agent_info = agent_policy.get_action(obs_option)

        obs, reward, dones, info = env.step(action)
        img = env.render(mode='rgb_array')
        frames.append(img)

        traj_list["observation"].append(obs)
        info['x'], info['y'] = obs[0], obs[1]
        traj_list["info"].append(info)
        if 'psi' in eval_type:
            Repr_obs_list.append(Psi(phi_obs_).detach().cpu().numpy()[0])
        else:
            Repr_obs_list.append(phi_obs_.detach().cpu().numpy()[0])    
        Obs_list.append(obs[:2])

        obs = torch.tensor(obs).unsqueeze(0).to(device).float()

    All_trajs_list.append(traj_list)
    All_Repr_obs_list.append(Repr_obs_list)

    if save_gif:
        gif_name = filepath + str(i) + '.gif'
        imageio.mimsave(gif_name, frames, 'GIF', duration=1)
        print('saved', gif_name)


Obs_array = np.array(Obs_list)
Repr_obs_array = np.array(All_Repr_obs_list)
Repr_goal_array = np.array(All_Repr_goal_list)


plot_trajectories(env, All_trajs_list, fig, ax1)
ax1.scatter(GoalList[:,0], GoalList[:,1])
colors = cm.rainbow(np.linspace(0, 1, len(All_Repr_obs_list)))
for i in range(Repr_obs_array.shape[0]):
    ax2.scatter(Repr_obs_array[i,:,0], Repr_obs_array[i,:,1], color=colors[i], s=5)
    ax2.scatter(Repr_goal_array[:,0], Repr_goal_array[:,1], marker='*', color=colors)

plt.savefig('lm.png')
print('saved at ', './lm.png')


