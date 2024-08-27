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
policy_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/baselinesd20230823_1724380008_ant_maze_metra/option_policy48000.pt"
traj_encoder_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/baselinesd20230823_1724380008_ant_maze_metra/traj_encoder48000.pt"
# # SGN-A
# policy_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/SGN_A/option_policy50000.pt"
# traj_encoder_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/SGN_A/traj_encoder50000.pt"

load_option_policy_base = torch.load(policy_path)
load_traj_encoder_base = torch.load(traj_encoder_path)
agent_policy = load_option_policy_base['policy'].eval()
agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()

# set up env
env = MazeWrapper("antmaze-medium-diverse-v0", random_init=False)
env.reset()
frames = []
fig, ax = plt.subplots()
np_random = np.random.default_rng(seed=0) 
goal = env.env.goal_sampler(np_random)
env.draw(ax)
goal_list = []
init_obs = env.reset()  

# settings:
Eval = 1
RandomInit = 0
num_goals = 1
max_path_length = 300
device = 'cuda'
# model_name = 'SGN-A'
model_name = 'baseline'
path = './test/' + model_name


FinallDistanceList = []
All_Repr_obs_list = []
All_Goal_obs_list = []
All_Return_list = []
All_GtReturn_list = []
All_trajs_list = []
FinallDistanceList = []
ArriveList=[]

with torch.no_grad():
    for i in range(num_goals):
        GoalList = env.env.goal_sampler(np_random)
        for j in trange(len(GoalList)):
            goal = GoalList[j]
            # print(goal)
            # get goal
            goal_list.append(goal)
            ax.scatter(goal[0], goal[1], s=25, marker='o', alpha=1, edgecolors='black')
            if Eval == 0:
                continue
            tensor_goal = torch.tensor(goal).to('cuda')
            # get obs
            if RandomInit:
                # to do:
                pass
            else:
                obs = env.reset()
            
            obs = torch.tensor(obs).unsqueeze(0).to(device).float()
            target_obs = env.get_target_obs(obs, tensor_goal)
            phi_target_obs = agent_traj_encoder(target_obs).mean
            phi_obs_ = agent_traj_encoder(obs).mean
            Repr_obs_list = []
            Repr_goal_list = []
            gt_return_list = []
            traj_list = {}
            traj_list["observation"] = []
            traj_list["info"] = []
            for t in range(max_path_length):
                option, phi_obs_, phi_target_obs = gen_z(target_obs, obs, traj_encoder=agent_traj_encoder, device=device, ret_emb=True)
                obs_option = torch.cat((obs, option), -1).float()
                # for viz
                Repr_obs_list.append(phi_obs_.cpu().numpy()[0])
                Repr_goal_list.append(phi_target_obs.cpu().numpy()[0])
                # get actions from policy
                action, agent_info = agent_policy.get_action(obs_option)
                # interact with the env
                obs, reward, dones, info = env.step(action)
                gt_dist = np.linalg.norm(goal - obs[:2])
                # for recording traj.2
                traj_list["observation"].append(obs)
                info['x'], info['y'] = env.env.get_xy()
                traj_list["info"].append(info)
                # calculate the repr phi
                obs = torch.tensor(obs).unsqueeze(0).to(device).float()
                gt_reward = - gt_dist / (30 * max_path_length)
                gt_return_list.append(gt_reward)
                

            All_Repr_obs_list.append(Repr_obs_list)
            All_Goal_obs_list.append(Repr_goal_list)
            All_GtReturn_list.append(gt_return_list)
            All_trajs_list.append(traj_list)
            FinallDistanceList.append(-gt_dist)
            if -gt_dist > -1:
                ArriveList.append(1)
            else:
                ArriveList.append(0)

# plot: goals  
filepath = path + '-cover_goals.png'
plt.savefig(filepath)
print("save:", filepath)


if Eval != 0:
    # calculate metrics
    FD = np.array(FinallDistanceList).mean()
    AR = np.array(ArriveList).mean()
    print("FD:", FD, '\n', "AR:", AR)

    # plot: traj.
    plot_trajectories(env, All_trajs_list, fig, ax)
    # ax.legend(loc='lower right')
    filepath = path + "-Maze_traj.png"
    plt.savefig(filepath) 
    print(filepath)
    # plot: repr_traj.
    PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length)
    print('Repr_Space_traj saved')



