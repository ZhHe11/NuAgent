'''
to test coverarge of maze;
'''
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio

from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze, plot_trajectories, plot_value
import d4rl
import torch
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from tqdm import trange, tqdm
import tests.make_env as make_env


# save the traj. as fig
def PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=100, is_PCA=False):
    Repr_obs_array = np.array(All_Repr_obs_list[0])
    All_Goal_obs_array = np.array(All_Goal_obs_list[0])
    for i in range(1,len(All_Repr_obs_list)):
        Repr_obs_array = np.concatenate((Repr_obs_array, np.array(All_Repr_obs_list[i])), axis=0)
        All_Goal_obs_array = np.concatenate((All_Goal_obs_array, np.array(All_Goal_obs_list[i])), axis=0)
    # 创建 PCA 对象，指定降到2维
    if is_PCA:
        pca = PCA(n_components=2)
        # 对数据进行 PCA
        Repr_obs_2d = pca.fit_transform(Repr_obs_array)
    else:
        Repr_obs_2d = Repr_obs_array
        All_Goal_obs_2d = All_Goal_obs_array
    # 绘制 PCA 降维后的数据
    plt.figure(figsize=(8, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(All_Repr_obs_list)))
    for i in range(0,len(All_Repr_obs_list)):
        color = colors[i]
        start_index = i * path_len
        end_index = (i+1) * path_len
        plt.scatter(Repr_obs_2d[start_index:end_index, 0], Repr_obs_2d[start_index:end_index, 1], color=color, s=5)
        plt.scatter(All_Goal_obs_2d[start_index:end_index, 0], All_Goal_obs_2d[start_index:end_index, 1], color=color, s=100, marker='*', edgecolors='black')
    path_file_traj = path + "-traj.png"
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('traj. in representation space')
    # plt.legend()
    plt.savefig(path_file_traj)


def vec_norm(vec):
    return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)


def gen_z(sub_goal, obs, traj_encoder, device="cpu", ret_emb: bool = False):
    goal_z = traj_encoder(sub_goal).mean
    target_cur_z = traj_encoder(obs).mean

    z = vec_norm(goal_z - target_cur_z)
    if ret_emb:
        return z, target_cur_z, goal_z
    else:
        return z

## load model
# baseline 
# policy_path = '/data/zh/project12_Metra/METRA/exp/ant/baselinesd000_1725971140_ant_metra/option_policy10000.pt'
# traj_encoder_path = '/data/zh/project12_Metra/METRA/exp/ant/baselinesd000_1725971140_ant_metra/traj_encoder10000.pt'

policy_path = "/data/zh/project12_Metra/METRA/exp/Debug/t01/option_policy20000.pt"
traj_encoder_path = "/data/zh/project12_Metra/METRA/exp/Debug/t01/traj_encoder20000.pt"

load_option_policy_base = torch.load(policy_path)
load_traj_encoder_base = torch.load(traj_encoder_path)
agent_policy = load_option_policy_base['policy'].eval()
agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()

# set up env
from envs.mujoco.ant_env import AntEnv
from iod.utils import get_normalizer_preset
from garagei.envs.consistent_normalized_env import consistent_normalize

env = AntEnv(render_hw=100)
normalizer_name = 'ant'
normalizer_kwargs = {}
normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

obs = env.reset()
obs_init = obs
frames = []
device = 'cuda'
# option = [-1, 0]
# option = torch.tensor(option).to(device).float().unsqueeze(0)
# goal = obs
# goal[:2] = [50,50]
# goal_tensor = torch.tensor(goal).to(device).float().unsqueeze(0)

num_eval = 5
num_model = 2
max_path_length = 200

all_distane = []
all_success = []
all_goalxy = []

goal_list = []
option_list = []
for i in range(num_eval):
    goal_xy = np.random.uniform(-100, 100, 2)
    goal = obs
    goal[:2] = goal_xy
    goal_list.append(goal)
    all_goalxy.append(goal_xy)
    option = np.random.uniform(-1, 1, 2)
    option = vec_norm(torch.tensor(option).to(device).float().unsqueeze(0))
    option_list.append(option)
    
obs_cover_range_list = []

for type in range(num_model):
    if type == 0:
        policy_path = '/data/zh/project12_Metra/METRA/exp/ant/baselinesd000_1725971140_ant_metra/option_policy10000.pt'
        traj_encoder_path = '/data/zh/project12_Metra/METRA/exp/ant/baselinesd000_1725971140_ant_metra/traj_encoder10000.pt'
    elif type == 1:
        policy_path = "/data/zh/project12_Metra/METRA/exp/Debug/t01/option_policy20000.pt"
        traj_encoder_path = "/data/zh/project12_Metra/METRA/exp/Debug/t01/traj_encoder20000.pt"
    
    load_option_policy_base = torch.load(policy_path)
    load_traj_encoder_base = torch.load(traj_encoder_path)
    agent_policy = load_option_policy_base['policy'].eval()
    agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()
    
    for i in trange(num_eval):
        obs = env.reset()
        obs_init = obs
        # goal_tensor = torch.tensor(goal_list[i]).to(device).float().unsqueeze(0)
        distance = []
        success = 0
        for t in range(max_path_length):
            obs_tensor = torch.tensor(obs).to(device).float().unsqueeze(0)
            phi_s = agent_traj_encoder(obs_tensor).mean
            # goal = obs
            # goal[:2] = goal_list[i][:2]
            # goal_tensor = torch.tensor(goal).to(device).float().unsqueeze(0)
            # phi_g = agent_traj_encoder(goal_tensor).mean
            # option = vec_norm(phi_g - phi_s)
            option = option_list[i]
            obs_option = torch.cat((obs_tensor, option), -1).float()
            action = agent_policy(obs_option)[1]['mean']
            
            obs, reward, done, info = env.step(action.detach().cpu().numpy())
            # print(obs)
            # print(info['coordinates'])
            obs_cover_range_list.append(obs)
            
            frames.append(env.render(mode='rgb_array'))
            distance.append(np.linalg.norm(obs[:2] - obs_init[:2]))
            # if distance[-1] < 0.1:
            #     success = 1
        all_distane.append(distance)
        # all_success.append(success)
        # all_goalxy.append(goal_xy)

# np_as = np.array(all_success)
np_ad = np.array(all_distane)

# print('average_success:', np_as.mean())
print('average_distance:', np_ad[:,-1].mean())

# plot curve
# import pdb; pdb.set_trace()
# x_features = range(len(obs))
# y1 = obs_cover_range_list[:num_eval*max_path_length] 
# y2 = obs_cover_range_list[num_eval*max_path_length:]

x_path = range(max_path_length)
for i in range(num_eval):
    label1 = 'baseline'
    label2 = 'ours'
    
    plt.plot(x_path, all_distane[i], label=label1, color='grey')
    plt.plot(x_path, all_distane[i+num_eval], label=label2, color='green')
    
    # for t in trange(len(y1)):
    #     plt.scatter([x_-0.125 for x_ in x_features], y1[t], color='grey', s=1)
    #     plt.scatter([x_+0.125 for x_ in x_features], y2[t], color='green', s=1)
# plt.xlabel('features')
# plt.ylabel('value')
# plt.title('cover ranges')

plt.xlabel('steps')
plt.ylabel('distance from init')
plt.title('distance curve')

plt.legend()
plt.savefig('./test/distance_curve.png')

# save the traj. as gif
# path = './test/'
# path_file = path + "ant_test.gif"
# imageio.mimsave(path_file, frames, duration=0.1)





