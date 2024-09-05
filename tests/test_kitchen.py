import sys
import os
os.environ["MUJOCO_GL"] = "osmesa"
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset

import numpy as np
import matplotlib.pyplot as plt
import imageio

import lexa.d4rl as d4rl
import torch
from sklearn.decomposition import PCA
import matplotlib.cm as cm

import imageio
from tqdm import trange, tqdm

local_lexa_path = os.path.abspath('./lexa')
sys.path.insert(0, local_lexa_path)
from envs.lexa.mykitchen import MyKitchenEnv
env = MyKitchenEnv(log_per_goal=True)
from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
env = FrameStackWrapper(env, 3)

# funtions
def vec_norm(vec):
    return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)
    
# 加载模型
# baseline
load_option_policy_base = torch.load("/data/zh/project12_Metra/METRA/exp/Debug/sd000_1725467783_kitchen_metra/wandb/run-20240904_163624-487z1jec/filesoption_policy.pt")
load_traj_encoder_base = torch.load("/data/zh/project12_Metra/METRA/exp/Debug/sd000_1725467783_kitchen_metra/wandb/run-20240904_163624-487z1jec/filestaregt_traj_encoder.pt")

# # ours
# load_option_policy_base = torch.load("/data/zh/project12_Metra/METRA/exp/kitchen/Target_traj_phi_learning-discrete-sd000_1725467722_kitchen_causer/wandb/latest-run/filesoption_policy.pt")
# load_traj_encoder_base = torch.load("/data/zh/project12_Metra/METRA/exp/kitchen/Target_traj_phi_learning-discrete-sd000_1725467722_kitchen_causer/wandb/latest-run/filestaregt_traj_encoder.pt")

policy = load_option_policy_base['policy']
traj_encoder = load_traj_encoder_base['target_traj_encoder']

# settings：
max_path_length = 50
option_dim = load_option_policy_base['dim_option']
path = '/data/zh/project12_Metra/METRA/tests/videos/local_test/'
Given_g = False
PhiPlot = True
num_task = 6

# 初始化
obs = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
Trajs = []
device = 'cuda'

# 加载goal
support_options = torch.eye(option_dim).to(device)
metric_success_task_relevant = {}
metric_success_all_objects = {}
if Given_g:
    all_goal_obs = []
    for i in range(num_task):
        goal_obs = env.render_goal(i)
        all_goal_obs.append(goal_obs)
        metric_success_task_relevant[i] = 0
        metric_success_all_objects[i] = 0
    all_goal_obs_tensor = torch.tensor(all_goal_obs, dtype=torch.float)
    eval_times = num_task
else:
    if option_dim == 2:
        directions = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
        support_options = torch.tensor(directions).to(device)
    eval_times = support_options.shape[0]

# interact with env
for i in trange(eval_times):
    # 初始化
    obs = env.reset()
    frames = []
    obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
    if Given_g:
        goal_tensor = torch.tile(all_goal_obs_tensor[i].reshape(-1), (3,1)).reshape(-1).unsqueeze(0).to('cuda')
        phi_g = traj_encoder(goal_tensor).mean
    else:
        support_option = support_options[i].unsqueeze(0)
    if PhiPlot: 
        Traj = []
        if Given_g:
            Traj.append(phi_g)
    # 每一条轨迹
    for t in trange(max_path_length):
        # policy inference:
        phi_s = traj_encoder(obs_tensor).mean
        if Given_g: 
            # to do; 需要映射；
            option = vec_norm(phi_g - phi_s)
        else:
            option = support_option
        obs_option = torch.cat((obs_tensor, option), -1).float()
        action_tensor = policy(obs_option)[1]['mean']
        action = action_tensor[0].detach().cpu().numpy()
        # iteration:
        obs, reward, _, info = env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
        # for viz:
        obs_img = info['image']
        frames.append(obs_img)
        if PhiPlot:
            Traj.append(phi_s)
            
    # metrics:
    success = np.zeros(num_task)
    for id_task in range(num_task):
        success[id_task] = env.compute_success(id_task)[0]
    print('success', success, 'option', option)
    # save traj. as gif
    gif_name = path+ str(i) + '.gif'
    imageio.mimsave(gif_name, frames, 'GIF', duration=1)
    print('saved', gif_name)
    Trajs.append(Traj)
    
# PCA to 2 dim for viz
Trajs_array = torch.concatenate([torch.concatenate(i) for i in Trajs]).detach().to('cpu').numpy()    # [batch*time, dim]
with open (path + 'Trajs.npy', 'wb') as f:
    np.save(f, Trajs_array)
Trajs_array_2dim = PCA(n_components=2).fit_transform(Trajs_array)    # [batch*time, 2]
Trajs_array_2dim = Trajs_array_2dim.reshape(len(Trajs), max_path_length, 2)    # [batch, time, 2]
colors = cm.rainbow(np.linspace(0, 1, len(Trajs)))
plt.figure()
for i in range(len(Trajs)):
    plt.plot(Trajs_array_2dim[i,:,0], Trajs_array_2dim[i,:,1], color=colors[i], label=str(i))
plt.legend()
plt.savefig(path + 'phi_traj.png')


