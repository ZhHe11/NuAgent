from envs.mujoco.ant_env import AntEnv
from iod.utils import get_normalizer_preset
from garagei.envs.consistent_normalized_env import consistent_normalize
from make_env import make_env_wo_args
from argparse import Namespace
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

def vec_norm(vec):
    return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)

def Psi(phi_x, max_path_length=300):
    return torch.tanh(2/max_path_length * (phi_x))

def load_models(model_path, epoch_num):
    policy_path = model_path + "/wandb/latest-run/filesoption_policy-" + str(epoch_num) + ".pt"
    traj_encoder_path = policy_path.replace('option_policy', 'traj_encoder')
    load_option_policy_base = torch.load(policy_path)
    load_traj_encoder_base = torch.load(traj_encoder_path)
    agent_policy = load_option_policy_base['policy'].eval()
    agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()
    
    return agent_policy, agent_traj_encoder

@torch.no_grad()
def InteractWithEnv(env_warp, agent_policy, agent_traj_encoder, **kwargs):
    env = env_warp
    args = Namespace(**kwargs)
    
    All_trajs_list = []
    All_Repr_obs_list = []
    All_Repr_goal_list = []
    All_DistacneList = []
    FinallDistanceList = []
    ArriveList = []
    All_Cover_list = []
    
    # decide the eval type:
    if args.eval_type in ['goal']:
        # load or generate
        GoalList = np.load('/mnt/nfs2/zhanghe/NuAgent/tests/savenp/less-LargeMazeGoal.npy')
        num_eval = len(GoalList)
    elif args.eval_type in ['random', 'uniform']:
        num_eval = args.num_eval
        random_options = np.random.uniform(-1,1, (num_eval, args.dim_option))
        random_options = torch.tensor(random_options).to(device)
        GoalList = []

    elif args.eval_type in ['random_goal']:
        num_eval = args.num_eval
        # you can adjust the range here
        GoalList = [np.random.uniform(-50, 50, 2) for i in range(num_eval)]
        
    else:
        num_eval = args.num_eval
        random_options = np.random.uniform(-1,1, (num_eval, args.dim_option))
        random_options = torch.tensor(random_options).to(device)
        GoalList = []
        
    # begin the eval for each option 
    for i in range(num_eval):
        obs = env.reset()
        obs_tensor = torch.tensor(obs).to(device).float().unsqueeze(0)
        phi_s = agent_traj_encoder(obs_tensor).mean
        if len(GoalList) > 0:
            goal = copy.deepcopy(obs)
            goal[:2] = GoalList[i]
            goal_tensor = torch.tensor(goal).to(device).float().unsqueeze(0)
            phi_g = agent_traj_encoder(goal_tensor).mean
            if args.psi:
                phi_g = Psi(phi_g, max_path_length=args.max_path_length)
                option = phi_g
            else:
                # baseline
                option = vec_norm(phi_g - phi_s)
                
        else: 
            goal = obs
            if args.eval_type in ['uniform']:
                option = random_options[i].unsqueeze(0)
            else:
                option = vec_norm(random_options[i]).unsqueeze(0)
            
        # before one traj
        All_Repr_goal_list.append(option.detach().cpu().numpy()[0])
        traj_list, Cover_list = {}, {}
        traj_list["observation"], traj_list["info"],  Obs_list, Repr_obs_list, frames = [], [], [], [], []
        DistacneList = []
        obs = torch.tensor(obs).unsqueeze(0).to(device).float()
        
        # begin one traj
        for t in trange(args.max_path_length):
            phi_obs_ = agent_traj_encoder(obs).mean
            obs_option = torch.cat((obs, option), -1).float()
            
            action, agent_info = agent_policy.get_action(obs_option)
            
            obs, reward, dones, info = env.step(action)
            gt_dist = np.linalg.norm(goal[:2] - obs[:2])
            
            # save video or not 
            if args.video:
                img = env.render(mode='rgb_array')
                frames.append(img)

            traj_list["observation"].append(obs)
            info['x'], info['y'] = obs[0], obs[1]
            traj_list["info"].append(info)
            DistacneList.append(gt_dist)
            
            if args.psi:
                Repr_obs_list.append(Psi(phi_obs_, max_path_length=args.max_path_length).detach().cpu().numpy()[0])
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
        
        All_trajs_list.append(traj_list)
        All_Repr_obs_list.append(Repr_obs_list)
        FinallDistanceList.append(gt_dist)
        Cover_list['env_infos']['coordinates'] = np.array(Cover_list['env_infos']['coordinates'])
        Cover_list['env_infos']['next_coordinates'] = np.array(Cover_list['env_infos']['next_coordinates'])
        All_Cover_list.append(Cover_list)
        All_DistacneList.append(DistacneList)
        
        if args.video:
            gif_name = args.SaveFilePath + str(i) + '.gif'
            imageio.mimsave(gif_name, frames, 'GIF', duration=1)
            print(f'video saved at {gif_name}')
    
    # after all geting traj 
    return {
        'All_trajs_list': np.array(All_trajs_list),
        'All_Repr_obs_list': np.array(All_Repr_obs_list),
        'All_Repr_goal_list': np.array(All_Repr_goal_list),
        'FinallDistanceList': np.array(FinallDistanceList),
        'All_Cover_list': np.array(All_Cover_list),
        'All_DistacneList': np.array(All_DistacneList),
        'GoalList': np.array(GoalList),
    }


def PlotRepr(Repr_obs_array, Repr_goal_array, colors, ax):
    ax.scatter(Repr_goal_array[:,0], Repr_goal_array[:,1], marker='*', color=colors)
    for i in range(Repr_obs_array.shape[0]):
        ax.scatter(Repr_obs_array[i,:,0], Repr_obs_array[i,:,1], color=colors[i], s=5)


def PlotObs(env, env_name, All_trajs_list, GoalList, fig, ax, colors):
    if 'maze' in env_name:
        from envs.AntMazeEnv import  plot_trajectories
        plot_trajectories(env, All_trajs_list, fig, ax, color_list=colors)
    elif env_name == 'ant':
        count = 0
        for color, trajectory in zip(colors, All_trajs_list):
            all_x = []
            all_y = []
            for info in trajectory['info']:
                all_x.append(info['x'])
                all_y.append(info['y'])
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            ax.scatter(all_x, all_y, s=5, c=color, alpha=0.2)
            ax.scatter(all_x[-1], all_y[-1], s=50, c=color, marker='*', alpha=1, edgecolors='black', label='traj.'+str(count))
            count += 1
    
    if len(GoalList) > 0:
        ax.scatter(GoalList[:,0], GoalList[:,1], s=5, c=colors, marker='o', alpha=1)
    
    return ax

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




if __name__ == '__main__':
    # settings
    kwargs = dict(
        env='ant',
        frame_stack=None,
        normalizer_type='preset',
        max_path_length=200,
        seed=0,    
        dim_option=2,
        eval_type='random',
        psi=False,
        video=False,
        SaveFilePath='./',
        num_eval=8,
    )
    # load models
    # model_path = "/mnt/nfs2/zhanghe/NuAgent/exp/ant/sd000_1731496075_ant_metra_bl"
    # model_path = '/mnt/nfs2/zhanghe/NuAgent/exp/ant/PSZP-unitsd000_1731587142_ant_PSZP'
    model_path = '/mnt/nfs2/zhanghe/NuAgent/exp/ant/sd000_1731496075_ant_metra_bl'
    kwargs['psi'] = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_num', '-e', type=int, default=1000)
    args = parser.parse_args()
    
    epoch_num = args.epoch_num
    
    agent_policy, agent_traj_encoder = load_models(model_path=model_path, epoch_num=epoch_num)
    
    # make env
    env_name = 'ant'
    env = make_env_wo_args(**kwargs)
    env.reset()

    # get option 
    device = 'cuda'
    
    # interact with env; and return traj(obs, repr) list, and metrics(distance, Coverage)
    Lists = InteractWithEnv(env, agent_policy, agent_traj_encoder, **kwargs)    
    
    # plot all traj
    colors = cm.rainbow(np.linspace(0, 1, len(Lists['All_Repr_obs_list'])))
    fig, axes = plt.subplots(1,2, figsize=(8,3))
    plt.suptitle(f'Epoch:{str(epoch_num)}')
    ## plot obs-space
    PlotObs(env, kwargs['env'], Lists['All_trajs_list'], Lists['GoalList'], fig, axes[0] , colors)
    ## plot repr-space 
    PlotRepr(Repr_obs_array=Lists['All_Repr_obs_list'], 
             Repr_goal_array=Lists['All_Repr_goal_list'],
             colors=colors, ax=axes[1])
    plt.savefig(model_path+'/TrajPlot.png')
    print(f"saved at {model_path+'/TrajPlot.png'}")
    
    # plot and save metrics as csv
    Lists['epoch'] = epoch_num
    ## Calculate Coverage
    eval_metrics = calc_eval_metrics(Lists['All_Cover_list'], is_option_trajectories=True)
    CoverCoords = eval_metrics['MjNumUniqueCoords']
    ## Calculate Distance
    FD = np.array(Lists['FinallDistanceList']).mean()
    print(f"FD: {FD}, CoverCoords, {CoverCoords}")
    ## save metrics as csv

    MetricDict = {
        'epoch': epoch_num,
        'FinallDistanceAvg': Lists['FinallDistanceList'].mean(),
        'FinallDistanceMin': Lists['FinallDistanceList'].min(),
        'CoverCoords': CoverCoords,
    }
    SaveMetrics(MetricDict, model_path=model_path, eval_type=kwargs['eval_type'])
    
    
    