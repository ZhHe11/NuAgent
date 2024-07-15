import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio

from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze, plot_trajectories, plot_value
import d4rl
import torch
from sklearn.decomposition import PCA
import matplotlib.cm as cm


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
        plt.scatter(Repr_obs_2d[start_index:end_index, 0], Repr_obs_2d[start_index:end_index, 1], color=color, s=5, label="traj."+str(i))
        # plt.scatter(All_Goal_obs_2d[start_index, 0], All_Goal_obs_2d[start_index, 1], marker='*', s=100, c=color, label="option."+str(i), edgecolors='black')
        plt.scatter(All_Goal_obs_2d[start_index:end_index, 0], All_Goal_obs_2d[start_index:end_index, 1], color=color, s=100, marker='*', label="option."+str(i), edgecolors='black')
    path_file_traj = path + "traj" + ".png"
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('traj. in representation space')
    plt.legend()
    plt.savefig(path_file_traj)


# setup the env
# env = GoalReachingMaze("antmaze-medium-diverse-v0")
env = MazeWrapper("antmaze-medium-diverse-v0")
env.reset()
frames = []
fig, ax = plt.subplots()

# goal 
# goals = env.four_goals()
# 随机Sample数据：
np_random = np.random.default_rng() 
goal = env.env.goal_sampler(np_random)
# print(goals)

env.draw(ax)

# # load model
load_option_policy_base = torch.load("/data/zh/project12_Metra/METRA/exp/Debug_ant_maze_baseline/sd000_1720752519_ant_maze_metra/option_policy155000.pt")
load_traj_encoder_base = torch.load("/data/zh/project12_Metra/METRA/exp/Debug_ant_maze_baseline/sd000_1720752519_ant_maze_metra/traj_encoder155000.pt")
load_option_policy = torch.load("/data/zh/project12_Metra/METRA/option_policy.pth")
load_traj_encoder = torch.load("/data/zh/project12_Metra/METRA/traj_encoder.pth")
# # load lastest model
load_option_policy_base['policy'].load_state_dict(load_option_policy)
load_traj_encoder_base['traj_encoder'].load_state_dict(load_traj_encoder)
# # eval mode
agent_policy = load_option_policy_base['policy'].eval()
agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()


# open the env, and set the init lists
device = 'cuda:0'
frames = []
All_Repr_obs_list = []
All_Goal_obs_list = []
All_Return_list = []
All_trajs_list = []
All_GtReturn_list = []

Pepr_viz = True
option_dim = 2

max_path_length = 500
goals_list = [
    [12.7, 16.5],
    [1.1, 12.9],
    [4.7, 4.5],
    [17.2, 0.9],
    [20.2, 20.1]
]
num_eval = len(goals_list)
goals = torch.tensor(np.array(goals_list)).to(device)

for i in range(num_eval):
    # 得到goal
    ax.scatter(goals_list[i][0], goals_list[i][1], s=50, marker='x', alpha=1, edgecolors='black', label='target.'+str(i))
    print(goals[i])
    
    obs = env.reset()  
    obs = torch.tensor(obs).unsqueeze(0).to(device).float()
    phi_obs_ = agent_traj_encoder(obs).mean
    
    Repr_obs_list = []
    Repr_goal_list = []
    option_return_list = []
    gt_return_list = []
    traj_list = {}
    traj_list["observation"] = []
    traj_list["info"] = []

    for t in range(max_path_length):
        # calculate the phi_obs
        phi_obs = phi_obs_
        
        # calculate the option:
        target_obs = env.get_target_obs(obs, goals[i])
        
        phi_target_obs = agent_traj_encoder(target_obs).mean
        option = phi_target_obs - phi_obs  
        # option = option / torch.norm(option, p=2)   
        
        # print("option", option)
        obs_option = torch.cat((obs, option), -1)
        
        # for viz
        if Pepr_viz:
            Repr_obs_list.append(phi_obs.cpu().detach().numpy()[0])
            Repr_goal_list.append(option.cpu().detach().numpy()[0])

        # get actions from policy
        action = agent_policy(obs_option)[1]['mean']

        # interact with the env
        obs, reward, done, info = env.step(action.cpu().detach().numpy()[0])
        gt_dist = np.linalg.norm(goals[i].cpu() - obs[:2])
        
        # # for recording traj.
        traj_list["observation"].append(obs)
        info['x'], info['y'] = env.env.get_xy()
        traj_list["info"].append(info)
        
        # calculate the repr phi
        obs = torch.tensor(obs).unsqueeze(0).to(device).float()
        phi_obs_ = agent_traj_encoder(obs).mean
        delta_phi_obs = phi_obs_ - phi_obs
        
        # option_reward and return
        option_reward = (option * delta_phi_obs).sum()
        option_return_list.append(option_reward.cpu().detach().numpy())
        gt_reward = - gt_dist / (30 * max_path_length)
        gt_return_list.append(gt_reward)
        
        # render for video
        frames.append(env.render(mode='rgb_array'))
        
    All_Repr_obs_list.append(Repr_obs_list)
    All_Goal_obs_list.append(Repr_goal_list)
    All_Return_list.append(option_return_list)
    All_trajs_list.append(traj_list)
    All_GtReturn_list.append(gt_return_list)

All_Return_array = np.array([np.array(i).sum() for i in All_Return_list])
All_GtReturn_array = np.array([np.array(i).sum() for i in All_GtReturn_list])
print(
    # "All_Return_array:", All_Return_array, '\n',
    "PolicyReturn:", All_Return_array.mean(), '\n', 
    "DistanceReturn", All_GtReturn_array.mean(),
)

# save the env as gif
# path = "/data/zh/project12_Metra/METRA/tests/videos/"
# path_file = path + "antmaze_test.gif"
# imageio.mimsave(path_file, frames, duration=1/24)
# print('video saved')

# plot_trajectories 用来绘制轨迹；
# print(trajs)
plot_trajectories(env, All_trajs_list, fig, ax)
# 保存图片；
path = "/data/zh/project12_Metra/METRA/tests/videos/"
ax.legend(loc='lower right')
plt.savefig(path + "maze.png")


if Pepr_viz:
    PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length)
    print('Repr_traj saved')


