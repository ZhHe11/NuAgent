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
import pickle

import argparse
import torch.distributions as dist


def UpdateGMM(dists, GMM=None, mix_dist_prob=None, device='cuda'):
    if GMM is None:
        component_distribution = dist.Independent(
            dist.Normal(
                loc=torch.stack([g.mean[0] for g in dists]),
                scale=torch.stack([g.stddev[0] for g in dists])
            ),
            reinterpreted_batch_ndims=1
        )
        if mix_dist_prob is None:
            # 创建均匀的 mixture_distribution
            mixture_distribution = dist.Categorical(
                probs=(torch.ones(len(dists)) / len(dists)).to(device)
            )
        else: 
            mixture_distribution = dist.Categorical(
                probs=mix_dist_prob
            )
        # 组合成一个 MixtureSameFamily 分布
        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )
        return window_dist

    else:
        component_distribution = GMM.component_distribution
        mixture_distribution = mixture_distribution

        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )

        return window_dist


def calc_eval_metrics(trajectories, is_option_trajectories, coord_dims=[0,1]):
    eval_metrics = {}
    coords = []
    for traj in trajectories:
        traj1 = traj['env_infos']['coordinates'][:, coord_dims]
        traj2 = traj['env_infos']['next_coordinates'][-1:, coord_dims]
        coords.append(traj1)
        coords.append(traj2)
    coords = np.concatenate(coords, axis=0)
    uniq_coords = np.unique(np.floor(coords), axis=0)
    eval_metrics.update({
        'MjNumUniqueCoords': len(uniq_coords),
    })
    return eval_metrics

@torch.no_grad()
def SaveMetrics(metrics, model_path, eval_type):
    import csv
    
    MetricsFilePath = model_path + '/'
    MetricsFile = MetricsFilePath + f'metrics_{eval_type}' + '.csv'
    
    if os.path.isfile(MetricsFile) and epoch_num != 0:
        with open(MetricsFile, 'r') as f:
            reader = csv.DictReader(f)
            existing_data = [row for row in reader]
            
    else:
        existing_data = []
    
    existing_data.append(metrics)
    
    with open(MetricsFile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing_data[0].keys())
        writer.writeheader()
        writer.writerows(existing_data)
    
    print(f'save at {MetricsFile}')


def vec_norm(vec):
    return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)


## def functions
def Psi(phi_x, phi_x0=None, max_path_length=300):
    return torch.tanh(2/max_path_length * (phi_x))

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch_num', type=int)
parser.add_argument('--path', type=str)
parser.add_argument('--eval_type', type=str)
args = parser.parse_args()

env = MazeWrapper("maze2d-large-v1", random_init=False)
obs = env.reset()
max_path_length=300

epoch_num = args.epoch_num
model_path = args.path
eval_type = args.eval_type
# model_path = '/mnt/nfs2/zhanghe/NuAgent/exp/LittleMaze/SZN-t01sd000_1731067339_lm_PSZP'
# model_path = '/mnt/nfs2/zhanghe/NuAgent/exp/LittleMaze/SZN-t05-no_g_dirsd000_1731067494_lm_PSZP'
paths = model_path + "/wandb/latest-run/filesoption_policy-" + str(epoch_num) + ".pt"
filepath = './'
policy_path = paths
traj_encoder_path = policy_path.replace('option_policy', 'traj_encoder')


load_option_policy_base = torch.load(policy_path)
load_traj_encoder_base = torch.load(traj_encoder_path)

agent_policy = load_option_policy_base['policy'].eval()
agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()

if 'psi' in args.eval_type:
    SZN_path =policy_path.replace('option_policy', 'SampleZPolicy')
    load_SZN_base = torch.load(SZN_path)
    window = load_SZN_base['window']


env.reset()
frames = []
num_eval = 50
dim_option = 2
device = 'cuda'
# eval_type = 'random_psi'
# method = 'baseline'
save_gif = False

fig, axes = plt.subplots(1,2, figsize=(8,3))
ax1 = axes[0]
ax2 = axes[1]
np_random = np.random.default_rng(seed=0) 
GoalList = np.load('/mnt/nfs2/zhanghe/NuAgent/tests/savenp/less-LargeMazeGoal.npy')

random_options = np.random.uniform(-1,1, (num_eval, dim_option))

All_trajs_list = []
All_Repr_obs_list = []
All_Repr_goal_list = []
FinallDistanceList = []
ArriveList=[]
All_Cover_list = []

if 'goal' in eval_type:
    num_eval = len(GoalList)

for i in trange(num_eval):
    Repr_obs_list = []
    Obs_list = []
    obs = env.reset()
    obs = torch.tensor(obs).unsqueeze(0).to(device).float()
    phi_obs_ = agent_traj_encoder(obs).mean
    
    # getting key;
    if eval_type in ['random', 'random_psi', 'uniform'] :
        option = torch.tensor(random_options[i]).unsqueeze(0).to(device)
        if 'uniform' not in eval_type:
            option = vec_norm(option)
        goal = np.zeros((num_eval, 2))
    
    elif eval_type in ['window_psi']:
        window_dist = UpdateGMM(window, device=device)
        option = window_dist.sample((1,))
        option = vec_norm(option)
        goal = np.zeros((num_eval, 2))
    
    elif 'goal' in eval_type:
        goal = GoalList[i]
        tensor_goal = torch.tensor(goal).to(device)
        obs_goal = copy.deepcopy(obs)
        obs_goal = env.get_target_obs(obs_goal, tensor_goal)
        Phi_g = agent_traj_encoder(obs_goal).mean
        if 'psi' in eval_type:
            option = Psi(Phi_g)
        else:
            option = vec_norm(Phi_g - phi_obs_)
        
    All_Repr_goal_list.append(option.detach().cpu().numpy()[0])
    traj_list = {}
    traj_list["observation"] = []
    traj_list["info"] = []
    frames = []
    Cover_list = {}
    arr = 0
    for t in trange(max_path_length):
        if arr == 0:
            # obs_goal = copy.deepcopy(obs)
            # obs_goal = env.get_target_obs(obs_goal, tensor_goal)
            # Phi_g = agent_traj_encoder(obs_goal).mean
            # if 'psi' in eval_type:
            #     option = Psi(Phi_g)
                
            phi_obs_ = agent_traj_encoder(obs).mean
            obs_option = torch.cat((obs, option), -1).float()
            
            action, agent_info = agent_policy.get_action(obs_option)
            
            obs, reward, dones, info = env.step(action)
            gt_dist = np.linalg.norm(goal - obs[:2])
            # img = env.render(mode='rgb_array')
            # frames.append(img)

            traj_list["observation"].append(obs)
            info['x'], info['y'] = obs[0], obs[1]
            traj_list["info"].append(info)
            if 'psi' in eval_type:
                Repr_obs_list.append(Psi(phi_obs_).detach().cpu().numpy()[0])
            else:
                Repr_obs_list.append(phi_obs_.detach().cpu().numpy()[0])    
            Obs_list.append(obs[:2])

            if 'env_infos' not in Cover_list:
                Cover_list['env_infos'] = {}
                Cover_list['env_infos']['coordinates'] = []
                Cover_list['env_infos']['next_coordinates'] = []
            Cover_list['env_infos']['coordinates'].append(obs[:2])
            Cover_list['env_infos']['next_coordinates'].append(obs[:2])
            
            obs = torch.tensor(obs).unsqueeze(0).to(device).float()
            ## if early stop:
            # gt_reward = - gt_dist
            # if gt_dist < 5:
            #     arr = 1
            
        # gt_return_list.append(gt_reward)
        else:
            info['x'], info['y'] = obs[0][0], obs[0][1]
            traj_list["info"].append(info)
            if 'psi' in eval_type:
                Repr_obs_list.append(Psi(phi_obs_).detach().cpu().numpy()[0])
            else:
                Repr_obs_list.append(phi_obs_.detach().cpu().numpy()[0])    
            Obs_list.append(obs[:2])
                        
    # print(gt_dist)
    All_trajs_list.append(traj_list)
    All_Repr_obs_list.append(Repr_obs_list)
    FinallDistanceList.append(-gt_dist)
    Cover_list['env_infos']['coordinates'] = np.array(Cover_list['env_infos']['coordinates'])
    Cover_list['env_infos']['next_coordinates'] = np.array(Cover_list['env_infos']['next_coordinates'])
    All_Cover_list.append(Cover_list)
    
    if arr:
        ArriveList.append(1)
    else:
        ArriveList.append(0)
    
    if save_gif:
        gif_name = filepath + str(i) + '.gif'
        imageio.mimsave(gif_name, frames, 'GIF', duration=1)
        print('saved', gif_name)


FD = np.array(FinallDistanceList).mean()
AR = np.array(ArriveList).mean()
eval_metrics = calc_eval_metrics(All_Cover_list, is_option_trajectories=True)
CoverCoords = eval_metrics['MjNumUniqueCoords']
print(f"FD: {FD}, AR:, {AR}, CoverCoords, {CoverCoords}")
metrics = {'epoch': epoch_num, 'FD': FD, 'AR': AR, 'CoverCoords': CoverCoords}
SaveMetrics(metrics, model_path, eval_type)

Obs_array = np.array(Obs_list)
Repr_obs_array = np.array(All_Repr_obs_list)
Repr_goal_array = np.array(All_Repr_goal_list)

colors = cm.rainbow(np.linspace(0, 1, len(All_Repr_obs_list)))
plot_trajectories(env, All_trajs_list, fig, ax1, color_list=colors)
# ax1.scatter(GoalList[:,0], GoalList[:,1], color=colors)
for i in range(Repr_obs_array.shape[0]):
    ax2.scatter(Repr_obs_array[i,:,0], Repr_obs_array[i,:,1], color=colors[i], s=5)
    ax2.scatter(Repr_goal_array[:,0], Repr_goal_array[:,1], marker='*', color=colors)


plt.savefig('./lm' + str(epoch_num) + '.png')
print('saved at ', './lm' + str(epoch_num) + '.png')


