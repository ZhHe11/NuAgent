import sys
import os
# os.environ["MUJOCO_GL"] = "osmesa"
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
load_option_policy_base = torch.load("/data/zh/project12_Metra/METRA/exp/SGN/SGN_D_neg-len_encouragesd000_1724994434_kitchen_metra/wandb/latest-run/filesoption_policy.pt")
load_traj_encoder_base = torch.load("//data/zh/project12_Metra/METRA/exp/SGN/SGN_D_neg-len_encouragesd000_1724994434_kitchen_metra/wandb/latest-run/filestaregt_traj_encoder.pt")
policy = load_option_policy_base['policy']
traj_encoder = load_traj_encoder_base['target_traj_encoder']

# settings：
max_path_length = 50

# 初始化
obs = env.reset()
frames = []
obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
device = 'cuda'


# 加载goal
metric_success_task_relevant = {}
metric_success_all_objects = {}
all_goal_obs = []
for i in range(6):
    goal_obs = env.render_goal(i)
    all_goal_obs.append(goal_obs)
    metric_success_task_relevant[i] = 0
    metric_success_all_objects[i] = 0
all_goal_obs_tensor = torch.tensor(all_goal_obs, dtype=torch.float)

for i in range(all_goal_obs_tensor.shape[0]):
    goal_tensor = torch.tile(all_goal_obs_tensor[i].reshape(-1), (3,1)).reshape(-1).unsqueeze(0).to('cuda')
    
    for t in trange(max_path_length):
        # policy
        phi_s = traj_encoder(obs_tensor).mean
        phi_g = traj_encoder(goal_tensor).mean
        option = vec_norm(phi_g - phi_s)
        print('option', option)
        obs_option = torch.cat((obs_tensor, option), -1).float()
        action_tensor = policy(obs_option)[1]['mean']
        action = action_tensor[0].detach().cpu().numpy()
        
        # iteration
        obs, reward, _, info = env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
        
        # for viz
        obs_img = info['image']
        frames.append(obs_img)
        # for metrics
        k = 'metric_success_task_relevant/goal_'+str(i)
        metric_success_all_objects[i] = max(metric_success_all_objects[i], info[k])
        k = 'metric_success_all_objects/goal_'+str(i)   
        metric_success_all_objects[i] = max(metric_success_all_objects[i], info[k])
        
        print('success', env.compute_success(i))
    
    gif_name = '/data/zh/project12_Metra/METRA/tests/videos/local_test/' + str(i) + '.gif'
    imageio.mimsave(gif_name, frames, 'GIF', duration=1)
    print('saved', gif_name)
    
print('metric_success_task_relevant:', sum(metric_success_task_relevant.values()) / len(metric_success_task_relevant))
print('metric_success_all_objects:',  sum(metric_success_all_objects.values()) / len(metric_success_all_objects))


