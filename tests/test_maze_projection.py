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
import copy


# save the traj. as fig
def PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=100, is_PCA=False, is_goal=True):
    if len(All_Goal_obs_list) == 0:
        is_goal = False
    
    Repr_obs_array = np.array(All_Repr_obs_list[0])
    if is_goal:
        All_Goal_obs_array = np.array(All_Goal_obs_list[0])
    for i in range(1,len(All_Repr_obs_list)):
        Repr_obs_array = np.concatenate((Repr_obs_array, np.array(All_Repr_obs_list[i])), axis=0)
        if is_goal:
            All_Goal_obs_array = np.concatenate((All_Goal_obs_array, np.array(All_Goal_obs_list[i])), axis=0)
    # 创建 PCA 对象，指定降到2维
    if is_PCA:
        pca = PCA(n_components=2)
        # 对数据进行 PCA
        Repr_obs_2d = pca.fit_transform(Repr_obs_array)
    else:
        Repr_obs_2d = Repr_obs_array
        if is_goal:
            All_Goal_obs_2d = All_Goal_obs_array
    # 绘制 PCA 降维后的数据
    plt.figure(figsize=(8, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(All_Repr_obs_list)))
    for i in range(0,len(All_Repr_obs_list)):
        color = colors[i]
        start_index = i * path_len
        end_index = (i+1) * path_len
        plt.scatter(Repr_obs_2d[start_index:end_index, 0], Repr_obs_2d[start_index:end_index, 1], color=color, s=5)
        if is_goal:
            plt.scatter(All_Goal_obs_2d[start_index:end_index, 0], All_Goal_obs_2d[start_index:end_index, 1], color=color, s=100, marker='*', edgecolors='black')
    path_file_traj = path + "-traj.png"
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('traj. in representation space')
    # plt.legend()
    plt.savefig(path_file_traj)


def vec_norm(vec):
    return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)

def Psi(phi_x, phi_x0, k=2, max_path_length=100):
    return 2 * (torch.sigmoid(k * (phi_x - phi_x0) / max_path_length) - 0.5)
    # return torch.tanh(phi_x - phi_x0)

def norm(x, keepdim=False):
    return torch.norm(x, p=2, dim=-1, keepdim=keepdim)        

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
policy_path = "/mnt/nfs2/zhanghe/NuAgent/exp/Maze/P-Exp2-reward_w200sd000_1728571903_ant_maze_SZN_P/wandb/latest-run/filesoption_policy-15000.pt"
traj_encoder_path = "/mnt/nfs2/zhanghe/NuAgent/exp/Maze/P-Exp2-reward_w200sd000_1728571903_ant_maze_SZN_P/wandb/latest-run/filestaregt_traj_encoder-15000.pt"


# policy_path = '/mnt/nfs2/zhanghe/NuAgent/exp/Maze/baselinesd000_1728470802_ant_maze_SZN_Z/option_policy11000.pt'
# traj_encoder_path = '/mnt/nfs2/zhanghe/NuAgent/exp/Maze/baselinesd000_1728470802_ant_maze_SZN_Z/traj_encoder11000.pt'

# policy_path = "/mnt/nfs2/zhanghe/NuAgent/exp/Maze/SZN-Exp4sd000_1728446621_ant_maze_SZN_Z/option_policy3000.pt"
# traj_encoder_path = "/mnt/nfs2/zhanghe/NuAgent/exp/Maze/SZN-Exp4sd000_1728446621_ant_maze_SZN_Z/traj_encoder3000.pt"

# # SGN-A
# policy_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/SGN_A/option_policy50000.pt"
# traj_encoder_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/SGN_A/traj_encoder50000.pt"
SZN_path = "/mnt/nfs2/zhanghe/NuAgent/exp/Maze/SZN-Exp4sd000_1728446621_ant_maze_SZN_Z/wandb/latest-run/filesSampleZPolicy.pt"


load_option_policy_base = torch.load(policy_path)
load_traj_encoder_base = torch.load(traj_encoder_path)
load_SZN_path_base = torch.load(SZN_path)
agent_policy = load_option_policy_base['policy'].eval()
if "target_traj_encoder" in load_traj_encoder_base.keys():
    agent_traj_encoder = load_traj_encoder_base['target_traj_encoder'].eval()
else:
    agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()
SZN = load_SZN_path_base['goal_sample_network'].eval()
input_token = load_SZN_path_base['input_token']
    
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
num_eval = 50
max_path_length = 100
device = 'cuda:0'
model_name = policy_path.split('/')[-4]
type = 'cover'
path = './test/' + 'Projection' + '-' + type
dim_option = 2


FinallDistanceList = []
All_Repr_obs_list = []
All_Goal_obs_list = []
All_Return_list = []
All_GtReturn_list = []
All_trajs_list = []
FinallDistanceList = []
ArriveList=[]

def eval_cover_rate(freq=1):
    with torch.no_grad():
        for i in range(num_goals):
            GoalList = env.env.goal_sampler(np_random, freq=freq)
            # GoalList = [
            #                     [12.7, 16.5],
            #                     [1.1, 12.9],
            #                     [4.7, 4.5],
            #                     [17.2, 0.9],
            #                     [20.2, 20.1],
            #                     [4.7, 0.9],
            #                     [0.9, 4.7],
            #                 ]
            
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
                obs = env.reset()
                obs = torch.tensor(obs).unsqueeze(0).to(device).float()
                obs_goal = copy.deepcopy(obs)
                obs_goal = env.get_target_obs(obs_goal, tensor_goal)
                phi_g = agent_traj_encoder(obs_goal).mean
                phi_obs_ = agent_traj_encoder(obs).mean
                option = Psi(phi_g, phi_obs_)
                
                Repr_obs_list = []
                Repr_goal_list = []
                gt_return_list = []
                traj_list = {}
                traj_list["observation"] = []
                traj_list["info"] = []
                for t in range(max_path_length):
                    phi_obs_ = agent_traj_encoder(obs).mean
                    obs_option = torch.cat((obs, option), -1).float()
                    # for viz
                    Repr_obs_list.append(phi_obs_.cpu().numpy()[0])
                    Repr_goal_list.append(phi_g.cpu().numpy()[0])
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


def eval_random_z(num_eval): 
    with torch.no_grad(): 
        options = np.random.randn(num_eval, dim_option)
        for i in trange(len(options)):
            obs = env.reset()
            option = torch.tensor(options[i]).unsqueeze(0).to(device)
            option = vec_norm(option)
            obs = torch.tensor(obs).unsqueeze(0).to(device).float()
            phi_obs_ = agent_traj_encoder(obs).mean
            Repr_obs_list = []
            gt_return_list = []
            traj_list = {}
            traj_list["observation"] = []
            traj_list["info"] = []
            for t in range(max_path_length):
                # option, phi_obs_, phi_target_obs = gen_z(target_obs, obs, traj_encoder=agent_traj_encoder, device=device, ret_emb=True)
                
                phi_obs_ = agent_traj_encoder(obs).mean
                obs_option = torch.cat((obs, option), -1).float()
                # for viz
                Repr_obs_list.append(phi_obs_.cpu().numpy()[0])
                # get actions from policy
                action, agent_info = agent_policy.get_action(obs_option)
                # interact with the env
                obs, reward, dones, info = env.step(action)
                # for recording traj.2
                traj_list["observation"].append(obs)
                info['x'], info['y'] = env.env.get_xy()
                traj_list["info"].append(info)
                # calculate the repr phi
                obs = torch.tensor(obs).unsqueeze(0).to(device).float()
                

            All_Repr_obs_list.append(Repr_obs_list)
            All_trajs_list.append(traj_list)


def viz_SZN_dist(num_sample=10):
    dist = SZN(input_token)
    # Data
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x,y)
    
    from scipy.stats import multivariate_normal

    for i in range(dist.mean.shape[0]):
        # Multivariate Normal
        mu_x = dist.mean[i][0].detach().cpu().numpy()
        sigma_x = dist.stddev[i][0].detach().cpu().numpy()
        mu_y = dist.mean[i][1].detach().cpu().numpy()
        sigma_y = dist.stddev[i][1].detach().cpu().numpy()
        rv = multivariate_normal([mu_x, mu_y], [[sigma_x, 0], [0, sigma_y]])

        # Probability Density
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        pd = rv.pdf(pos)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, pd, cmap='viridis', linewidth=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')
        plt.title("Multivariate Normal Distribution")
        plt.savefig('test' + str(i) + '.png')
        plt.close()
        print(mu_x, mu_y, sigma_x, sigma_y)
    
    
def viz_psi_space(num_eval=10):
    with torch.no_grad(): 
        options = np.random.randn(num_eval, dim_option)
        for i in trange(len(options)):
            obs = env.reset()
            option = torch.tensor(options[i]).unsqueeze(0).to(device)
            # option = vec_norm(option)       # 可以不vec
            obs = torch.tensor(obs).unsqueeze(0).to(device).float()
            phi_obs_ = agent_traj_encoder(obs).mean
            phi_x0 = phi_obs_
            Repr_obs_list = []
            Repr_goal_list = []
            traj_list = {}
            traj_list["observation"] = []
            traj_list["info"] = []
            for t in range(max_path_length):
                
                phi_obs_ = agent_traj_encoder(obs).mean
                obs_option = torch.cat((obs, option), -1).float()
                psi_obs = Psi(phi_obs_, phi_x0, max_path_length=max_path_length)
                # for viz
                Repr_obs_list.append(psi_obs.cpu().numpy()[0])
                Repr_goal_list.append(option.cpu().numpy()[0])

                # get actions from policy
                action, agent_info = agent_policy.get_action(obs_option)
                # interact with the env
                obs, reward, dones, info = env.step(action)
                # for recording traj.2
                traj_list["observation"].append(obs)
                info['x'], info['y'] = env.env.get_xy()
                traj_list["info"].append(info)
                # calculate the repr phi
                obs = torch.tensor(obs).unsqueeze(0).to(device).float()
                
            All_Repr_obs_list.append(Repr_obs_list)
            All_Goal_obs_list.append(Repr_goal_list)
            All_trajs_list.append(traj_list)
    

if __name__ == '__main__':
    # exe:
    if type == 'cover':
        eval_cover_rate(freq=2)
        filepath = path + '-cover_goals.png'
        plt.savefig(filepath)
        print("save:", filepath)
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
        PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=True)
        print('Repr_Space_traj saved')


    elif type == 'random_z':
        eval_random_z(num_eval)
        # plot: traj.
        plot_trajectories(env, All_trajs_list, fig, ax)
        # ax.legend(loc='lower right')
        filepath = path + "-Maze_traj.png"
        plt.savefig(filepath) 
        print(filepath)
        # plot: repr_traj.
        PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=False)
        print('Repr_Space_traj saved')


    elif type == 'SZN_dist':
        viz_SZN_dist()
        
        
    elif type == 'viz_psi_space':
        viz_psi_space(50)
        filepath = path + '-cover_goals.png'
        plt.savefig(filepath)
        print("save:", filepath)
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
        PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=True)
        print('Repr_Space_traj saved')

