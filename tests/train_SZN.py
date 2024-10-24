
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

import torch.optim as optim
from iod.GradCLipper import GradClipper

# save the traj. as fig
def PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=100, is_PCA=False, is_goal=True):
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


def gen_z(sub_goal, obs, traj_encoder, device="cpu", ret_emb: bool = False):
    goal_z = traj_encoder(sub_goal).mean
    target_cur_z = traj_encoder(obs).mean

    z = vec_norm(goal_z - target_cur_z)
    if ret_emb:
        return z, target_cur_z, goal_z
    else:
        return z



from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians

def _get_concat_obs(obs, option):
    return get_torch_concat_obs(obs, option)


def Psi(phi_x):
    return torch.tanh(1/300 * phi_x)


def EstimateValue(policy, alpha, qf1, qf2, option, state, num_samples=1):
    '''
    num_samles越大,方差越小,偏差不会更小;
    '''
    batch = option.shape[0]
    # [s0, z]
    processed_cat_obs = _get_concat_obs(policy.process_observations(state), option.float())     # [b,dim_s+dim_z]
    
    # dist of pi(a|[s0, z])
    dist, info = policy(processed_cat_obs)    # [b, dim]
    actions = dist.sample((num_samples,))          # [n, b, dim]
    log_probs = dist.log_prob(actions).squeeze(-1)  # [n, b]
    
    processed_cat_obs_flatten = processed_cat_obs.repeat(num_samples, 1, 1).view(batch * num_samples, -1)      # [n*b, dim_s+z]
    actions_flatten = actions.view(batch * num_samples, -1)     # [n*b, dim_a]
    q_values = torch.min(qf1(processed_cat_obs_flatten, actions_flatten), qf2(processed_cat_obs_flatten, actions_flatten))      # [n*b, dim_1]
    
    alpha = alpha.param.exp()
        
    values = q_values - alpha * log_probs.view(batch*num_samples, -1)      # [n*b, 1]
    values = values.view(num_samples, batch, -1)        # [n, b, 1]
    E_V = values.mean(dim=0)        # [b, 1]

    
    return E_V.squeeze(-1)


def viz_dist_circle(window, path, psi_z=None):
    from matplotlib.patches import Ellipse
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    for i in range(len(window)):
        dist = window[i]
        for i in range(dist.mean.shape[0]):
            mu_x = dist.mean[i][0].detach().cpu().numpy()
            sigma_x = dist.stddev[i][0].detach().cpu().numpy()
            mu_y = dist.mean[i][1].detach().cpu().numpy()
            sigma_y = dist.stddev[i][1].detach().cpu().numpy()
            e = Ellipse(xy = (mu_x,mu_y), width = sigma_x * 2, height = sigma_y * 2, angle=0)
            ax.add_artist(e)
        
    if psi_z is not None:
        ax.scatter(psi_z[:, 0], psi_z[:, 1], marker='*', alpha=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(path + '-c' + '.png')
    print("save at:", path + '-c' + '.png')
    plt.close()



def norm(x, keepdim=False):
    return torch.norm(x, p=2, dim=-1, keepdim=keepdim)     

@torch.no_grad()
def viz_Value_in_Psi(policy, alpha, qf1, qf2, state, num_samples=10, device='cpu', path='./', label='1'):
    density = 200
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(18, 12), facecolor='w')
    
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pos = torch.tensor(pos).to(device)
    pos_flatten = pos.view(-1,2)
    option = pos_flatten
    state_batch = state.unsqueeze(0).repeat(option.shape[0], 1)

    V_flatten = EstimateValue(policy, alpha, qf1, qf2, option, state_batch, num_samples=100)
    V = V_flatten.view(pos.shape[0],pos.shape[1])
    
    print(V.max(), V.min())
    
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, V.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(60, 35)
    ax.set_xlabel('X')          
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    plt.savefig(path + '-Value' + '.png')
    print('save at: ' + path + '-Value' + label + '.png')
    plt.close()


@torch.no_grad()
def viz_Regert_in_Psi(base1, base2, state, num_samples=10, device='cpu', path='./'):
    def get_fuctions(base):
        return base['qf1'], base['qf2'], base['alpha'], base['policy'] 
    
    density = 100
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(18, 12), facecolor='w')
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pos = torch.tensor(pos).to(device)
    pos_flatten = pos.view(-1,2)
    option = pos_flatten
    state_batch = state.unsqueeze(0).repeat(option.shape[0], 1)
    
    # value 1:
    qf1, qf2, alpha, policy = get_fuctions(base1)
    V1 = EstimateValue(policy, alpha, qf1, qf2, option, state_batch, num_samples=num_samples)   # / (torch.clamp(norm(option), min=0.5))
    V1 = V1.view(pos.shape[0],pos.shape[1])
    
    # value 2:
    qf1, qf2, alpha, policy = get_fuctions(base2)
    V2 = EstimateValue(policy, alpha, qf1, qf2, option, state_batch, num_samples=num_samples)   # / (torch.clamp(norm(option), min=0.5))
    V2 = V2.view(pos.shape[0],pos.shape[1])
    
    # Regret:
    Regret = V2 - V1
    
    print(Regret.max(), Regret.min())
    
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, Regret.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(60, 35)
    ax.set_xlabel('X')          
    ax.set_ylabel('Y')
    ax.set_zlabel('Regret')
    plt.savefig(path + '-Regret' + '.png')
    print('save at: ' + path + '-Regret' + '.png')
    plt.close()
    
    # # Regret-scale:
    # Regret = (V2 - V1) / (1- (1-0.99)/2 * V1)
    # print("scale", (1- (1-0.99)/2 * V1))
    # print(Regret.max(), Regret.min())
    # fig = plt.figure(figsize=(18, 12), facecolor='w')
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.plot_surface(X, Y, Regret.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.view_init(60, 35)
    # ax.set_xlabel('X')          
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Regret-scale')
    # plt.savefig(path + '-Regret-scale' + '.png')
    # print('save at: ' + path + '-Regret-scale' + '.png')
    # plt.close()
    
    # plot Value:
    fig = plt.figure(figsize=(18, 12), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V1.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(60, 35)
    ax.set_xlabel('X')          
    ax.set_ylabel('Y')
    ax.set_zlabel('V1')
    plt.savefig(path + '-Value1' + '.png')
    print('save at: ' + path + '-Value1' + '.png')
    plt.close()
    
    # plot Value:
    fig = plt.figure(figsize=(18, 12), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V2.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(60, 35)
    ax.set_xlabel('X')          
    ax.set_ylabel('Y')
    ax.set_zlabel('V2')
    plt.savefig(path + '-Value2' + '.png')
    print('save at: ' + path + '-Value2' + '.png')
    plt.close()
    
    
    
    
    


## load model
# baseline 
policy_path = "/mnt/nfs2/zhanghe/NuAgent/exp/SaveModel/filesoption_policy-500.pt"
policy_path2 = "/mnt/nfs2/zhanghe/NuAgent/exp/SaveModel/filesoption_policy-600.pt"
traj_encoder_path = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PPAU-constraint7-uniform-3sd000_1729427323_ant_maze_P_SZN_AU/wandb/latest-run/filestaregt_traj_encoder-1000.pt"
# SZN_path = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PPAU-constraint7-uniform-3sd000_1729427323_ant_maze_P_SZN_AU/wandb/latest-run/filesSampleZPolicy-1000.pt"


SZN_path_0 = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PSZN-R4-Window-4sd000_1729659070_ant_maze_P_SZN_AU/wandb/latest-run/filesSampleZPolicy-0.pt"

# policy_path = "/mnt/nfs2/zhanghe/NuAgent/exp/Maze/SZN-Exp4sd000_1728446621_ant_maze_SZN_Z/option_policy3000.pt"
# traj_encoder_path = "/mnt/nfs2/zhanghe/NuAgent/exp/Maze/SZN-Exp4sd000_1728446621_ant_maze_SZN_Z/traj_encoder3000.pt"

# # SGN-A
# policy_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/SGN_A/option_policy50000.pt"
# traj_encoder_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/SGN_A/traj_encoder50000.pt"

load_option_policy_base = torch.load(policy_path)
load_option_policy_base2 = torch.load(policy_path2)
load_traj_encoder_base = torch.load(traj_encoder_path)
load_SZN_path_base = torch.load(SZN_path_0)
agent_policy = load_option_policy_base['policy'].eval()
if "target_traj_encoder" in load_traj_encoder_base.keys():
    agent_traj_encoder = load_traj_encoder_base['target_traj_encoder'].eval()
else:
    agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()
SZN = load_SZN_path_base['goal_sample_network'].eval()
input_token = load_SZN_path_base['input_token']
    
# set up env
env = MazeWrapper("antmaze-medium-diverse-v0", random_init=False)
obs0 = env.reset()
frames = []
# fig, ax = plt.subplots()
np_random = np.random.default_rng(seed=0) 
goal = env.env.goal_sampler(np_random)
# env.draw(ax)
goal_list = []
init_obs = env.reset()  

# settings:
Eval = 1
RandomInit = 0
num_goals = 1
num_eval = 50
max_path_length = 100
device = 'cuda'
model_name = policy_path.split('/')[-4]
path = './test/' + model_name
dim_option = 2
type = 'random_z'


s0 = torch.tensor(obs0).to(device).float()
psi_s0 = Psi(agent_traj_encoder(s0).mean)
viz_Regert_in_Psi(base1=load_option_policy_base, base2=load_option_policy_base2, state=s0, device=device, num_samples=10)
    
def get_fuctions(base):
    return base['qf1'], base['qf2'], base['alpha'], base['policy'] 

def norm(x, keepdim=False):
    return torch.norm(x, p=2, dim=-1, keepdim=keepdim)        

qf1, qf2, alpha, policy = get_fuctions(load_option_policy_base2)
last_qf1, last_qf2, last_alpha, last_policy = get_fuctions(load_option_policy_base)
grad_clip = GradClipper(clip_type='clip_norm', threshold=3, norm_type=2)

exit()

train_SZN = 1
DistWindow = [SZN(input_token)]
for i in range(10):
    # to do: 在SZN的loss中加入一个KL散度的loss，这个kl散度是当前z和window中的分布之间的kl散度；让kl散度尽可能增大，可以让window中的sample的方向尽可以能多，同时也能保证是Regert较大的分布；
    SampleZPolicy_optim = optim.Adam(SZN.parameters(), lr=1e-1)
    
    for t in range(50):
        # Reset the SZN:
        dist_z = SZN(input_token)
        z = dist_z.sample()
        z_logp = dist_z.log_prob(z)
        V_z = EstimateValue(policy=policy, alpha=alpha, qf1=qf1, qf2=qf2, option=z, state=s0.unsqueeze(0).repeat(z.shape[0], 1))
        
        V_z_last_iter = EstimateValue(policy=last_policy, alpha=last_alpha, qf1=last_qf1, qf2=last_qf2, option=z, state=s0.unsqueeze(0).repeat(z.shape[0], 1))

        Regret = V_z - V_z_last_iter
        # V_szn = (Regret - Regret.mean()) / (Regret.std() + 1e-3)\
        V_szn = Regret

        SampleZPolicy_optim.zero_grad()    
        w = 0.01
        Kl_sum = 0
        for i in range(len(DistWindow)):
            dist_i = DistWindow[i]
            log_qz = dist_i.log_prob(z)
            log_pz = z_logp
            pz = torch.exp(log_pz)
            Kl_sum += - pz * (log_pz - log_qz)
        
        kl_window = log_pz / len(DistWindow)
        
        loss_SZP = (-z_logp * V_szn - w * dist_z.entropy() - kl_window).mean()
        loss_SZP.backward()
        grad_clip.apply(SZN.parameters())
        SampleZPolicy_optim.step()
                
    # window queue operation    
    with torch.no_grad():
        dist = SZN(input_token)    
        is_different = 1
        for j in range(len(DistWindow)):
            dist_mean = dist.mean
            window_j_mean = DistWindow[j].mean
            if (norm(dist_mean - window_j_mean)).mean() < 0.1:
                is_different = 0
                break
        if is_different == 1:               
            DistWindow.append(dist)
            if len(DistWindow) > 10:
                DistWindow.pop(0)
                
    psi_g = SZN(input_token).sample().detach()

    # sample SZN from window
    random_index = np.random.randint(0, len(DistWindow))
    print(random_index)
    dist = DistWindow[random_index]
    last_z = dist.sample()

    np_z = last_z.cpu().numpy()
    print("Sample Z: ", np_z)

    from iod.P_SZN_AU import viz_SZN_dist_circle
    viz_SZN_dist_circle(SZN, input_token, path=path, psi_z=np_z)

viz_dist_circle(DistWindow, path=path+'window', psi_z=np_z)



exit()

FinallDistanceList = []
All_Repr_obs_list = []
All_Goal_obs_list = []
All_Return_list = []
All_GtReturn_list = []
All_trajs_list = []
FinallDistanceList = []
ArriveList=[]

def eval_cover_rate(freq=5):
    with torch.no_grad():
        for i in range(num_goals):
            GoalList = env.env.goal_sampler(np_random, freq=freq)
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
        PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=False)
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
        
        

