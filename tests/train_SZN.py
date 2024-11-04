
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

from torch.distributions import Normal, Categorical, MixtureSameFamily

from iod.viz_utils import viz_SZN_dist_circle, viz_dist_circle, viz_GMM_circle
import torch.distributions as dist



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
    Regret = V2
    
    print(Regret.max(), Regret.min())
    
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, Regret.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(60, 270+20)
    ax.set_xlabel('X')          
    ax.set_ylabel('Y')
    ax.set_zlabel('Regret')
    plt.savefig(path + '-Regret' + '.png')
    print('save at: ' + path + '-Regret' + '.png')
    plt.close()


## load model
# baseline 
import argparse
parser = argparse.ArgumentParser(description="A simple example of argparse usage")
parser.add_argument('-e', '--epoch_num', type=int, default=700)
args = parser.parse_args()

env = MazeWrapper("antmaze-medium-diverse-v0", random_init=False)
epoch_num = args.epoch_num
policy_path = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PSZP-8-GMM1-Deque10-std1_3e1_1e1sd000_1730290115_ant_maze_PSZP/wandb/latest-run/filesoption_policy-" + str(epoch_num) +'.pt'
policy_path1 = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PSZP-8-GMM1-Deque10-std1_3e1_1e1sd000_1730290115_ant_maze_PSZP/wandb/latest-run/filesoption_policy-" + str(epoch_num-100) +'.pt'
policy_path2 = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PSZP-8-GMM1-Deque10-std1_3e1_1e1sd000_1730290115_ant_maze_PSZP/wandb/latest-run/filesoption_policy-" + str(epoch_num-200) +'.pt'

traj_encoder_path = policy_path.replace("option_policy", "traj_encoder")
SZN_path = policy_path1.replace("option_policy", "SampleZPolicy")
SZN_path0 = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PSZP-8-GMM1-Deque10-std1_3e1_1e1sd000_1730290115_ant_maze_PSZP/wandb/latest-run/filesSampleZPolicy-0.pt"
# policy_path = "/mnt/nfs2/zhanghe/NuAgent/exp/Maze/SZN-Exp4sd000_1728446621_ant_maze_SZN_Z/option_policy3000.pt"
# traj_encoder_path = "/mnt/nfs2/zhanghe/NuAgent/exp/Maze/SZN-Exp4sd000_1728446621_ant_maze_SZN_Z/traj_encoder3000.pt"

# # SGN-A
# policy_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/SGN_A/option_policy50000.pt"
# traj_encoder_path = "/data/zh/project12_Metra/METRA/exp/Debug_baseline/SGN_A/traj_encoder50000.pt"

load_option_policy_base_k = torch.load(policy_path)
load_option_policy_base_kminus1 = torch.load(policy_path1)
load_traj_encoder_base = torch.load(traj_encoder_path)
load_SZN_path_base = torch.load(SZN_path)
if "target_traj_encoder" in load_traj_encoder_base.keys():
    agent_traj_encoder = load_traj_encoder_base['target_traj_encoder'].eval()
else:
    agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()
SZN = load_SZN_path_base['goal_sample_network']
ResetSZPolicy = torch.load(SZN_path0)['goal_sample_network']
input_token = load_SZN_path_base['input_token']
DistWindowList = load_SZN_path_base['window']
device = 'cuda'


def UpdateGMM(dists, GMM=None, device='cuda'):
    if GMM is None:
        component_distribution = dist.Independent(
            dist.Normal(
                loc=torch.stack([g.mean[0] for g in dists]),
                scale=torch.stack([g.stddev[0] for g in dists])
            ),
            reinterpreted_batch_ndims=1
        )

        # 创建均匀的 mixture_distribution
        mixture_distribution = dist.Categorical(
            probs=(torch.ones(len(dists)) / len(dists)).to(device)
        )

        # 组合成一个 MixtureSameFamily 分布
        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )

        return window_dist
    
    else:

        means_from_component = GMM.component_distribution.base_dist.loc
        stddevs_from_component = GMM.component_distribution.base_dist.scale

        # 队列方法更新GMM
        num_new = len(dists)
        window_len = len(means_from_component)

        means_tmp = torch.zeros_like(means_from_component).to(device)
        means_tmp[:window_len-num_new] = means_from_component[num_new:]
        stddev_tmp = torch.zeros_like(stddevs_from_component).to(device)
        stddev_tmp[:window_len-num_new] = stddevs_from_component[num_new:]
        for i in range(num_new):
            means_tmp[window_len-num_new+i] = dists[i].mean[0]
            stddev_tmp[window_len-num_new+i] = dists[i].stddev[0]
        

        component_distribution = dist.Independent(
            dist.Normal(
                loc=means_tmp,
                scale=stddev_tmp
            ),
            reinterpreted_batch_ndims=1
        )
        mixture_distribution = dist.Categorical(
            probs=(torch.ones(window_len) / window_len).to(device)
        )
        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )

        return window_dist


window_dist = UpdateGMM(DistWindowList)
viz_GMM_circle(window_dist, path='./1')

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
GMM = 1

ConfidenceFactor = 1
ReprBuffer = np.load("/mnt/nfs2/zhanghe/NuAgent/AnalysisData/PSZP-8-GMM1-Deque10-std1_3e1_1e1sd000_1730290115_ant_maze_PSZP-Repr_obs_list.npy")
SfReprBuffer = ReprBuffer[:,-1]



s0 = torch.tensor(obs0).to(device).float()
psi_s0 = Psi(agent_traj_encoder(s0).mean)
viz_Regert_in_Psi(base1=load_option_policy_base_kminus1, base2=load_option_policy_base_k, state=s0, device=device, num_samples=10)
    
def get_fuctions(base):
    return base['qf1'], base['qf2'], base['alpha'], base['policy'] 

def norm(x, keepdim=False):
    return torch.norm(x, p=2, dim=-1, keepdim=keepdim)        

qf1, qf2, alpha, policy = get_fuctions(load_option_policy_base_k)
last_qf1, last_qf2, last_alpha, last_policy = get_fuctions(load_option_policy_base_kminus1)
grad_clip = GradClipper(clip_type='clip_norm', threshold=3, norm_type=2)

# exit()
init_obs = s0.unsqueeze(0).repeat(input_token.shape[0], 1)
SampleZPolicy = SZN
train_SZN = 1

with torch.no_grad():
    DistWindow = DistWindowList

option_policy = policy
log_alpha = alpha
for i in range(10):
    SampleZPolicy_optim = optim.Adam(SZN.parameters(), lr=3e-2)

    def copy_params(ori_model, target_model):
        for t_param, param in zip(target_model.parameters(), ori_model.parameters()):
            t_param.data.copy_(param.data)

    copy_params(ResetSZPolicy, SampleZPolicy)

    for t in range(100):
        # Reset the SZN:

        dist_z = SampleZPolicy(input_token)
        ## Sample Z from dist_z
        z_repeat = dist_z.rsample((1,))
        ## [fatal bug !!!!!] if using rsmaple, z in log p must be detach, because logp owns grads itself.
        z_logp_repeat = dist_z.log_prob(z_repeat.detach())
        z = z_repeat.view(-1,dim_option)
        z_logp = z_logp_repeat.view(-1)
        
        V_z =  EstimateValue(policy= option_policy, alpha=log_alpha, qf1=qf1, qf2=qf2, option=z, state=s0.unsqueeze(0).repeat(z.shape[0], 1))
        V_z_last_iter = EstimateValue(policy=last_policy, alpha=last_alpha, qf1=last_qf1, qf2=last_qf2, option=z, state=s0.unsqueeze(0).repeat(z.shape[0], 1))

        V_szn = 0 * (V_z - V_z_last_iter) + V_z_last_iter
        # V_szn = (V_szn - V_szn.mean()) / (V_szn.std() + 1e-6)
    
        SampleZPolicy_optim.zero_grad()
        w1 = 0
        w2 = 3

        if GMM:
            log_pz = window_dist.log_prob(z)
            pz = torch.exp(log_pz)
            log_qz = z_logp
            kl_window = pz * (log_pz - log_qz)

        else:
            Kl_sum = 0
            for i in range(len(DistWindow)):
                dist_i = DistWindow[i]
                log_pz = dist_i.log_prob(z)
                pz = torch.exp(log_pz)
                log_qz = z_logp
                Kl_sum += pz * (log_pz - log_qz)
                
            if len(DistWindow) > 0:
                kl_window = Kl_sum / len(DistWindow)
            else:
                kl_window = torch.zeros(Kl_sum.shape).to(device)
        


        w3 = 0
        confidence = 0
        if ConfidenceFactor == 1:
            confidence = torch.clamp(torch.norm(z.unsqueeze(1) - torch.tensor(SfReprBuffer).to(device).unsqueeze(0), dim=-1).min(dim=-1)[0], min=0.1)

        print(confidence[0])
        V_szn = V_szn / confidence
        V_szn = (V_szn - V_szn.mean()) / (V_szn.std() + 1e-6)

            
        
        loss_SZP = (1 * -z_logp * V_szn.detach() - w1 * dist_z.entropy() - w2 * kl_window + w3 * confidence).mean()
        # loss_SZP = (-z_logp * V_szn).mean()
        loss_SZP.backward()
        grad_clip.apply(SampleZPolicy.parameters())
        SampleZPolicy_optim.step()


        # print(confidence.mean())

    # # window queue operation    
    with torch.no_grad():
        dist_SZN = SZN(input_token)    

        if GMM: 
            DistWindow = [dist_SZN]
            if len(DistWindow) > window_dist.component_distribution.base_dist.loc.shape[0]:
                DistWindow.pop(0)

        else: 
            is_different = 1
            for j in range(len(DistWindow)):
                dist_mean = dist_SZN.mean
                window_j_mean = DistWindow[j].mean
                if (norm(dist_mean - window_j_mean)).mean() < 0.1:
                    is_different = 0
                    break
            if is_different == 1:               
                DistWindow.append(dist_SZN)
                if len(DistWindow) > 10:
                    DistWindow.pop(0)

    if GMM:
        with torch.no_grad():
            window_dist = UpdateGMM(DistWindow, window_dist)
        viz_GMM_circle(window_dist, path='./1', psi_z=SfReprBuffer)

    psi_g = SZN(input_token).sample().detach()


    # sample SZN from window
    random_index = np.random.randint(0, len(DistWindow))
    if not GMM:
        viz_SZN_dist_circle(SZN, input_token, path=path)

if not GMM:
    viz_dist_circle(DistWindow, path=path+'window')




