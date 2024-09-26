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
from sklearn.manifold import TSNE
import matplotlib.cm as cm

import imageio

from tqdm import trange, tqdm

from envs.custom_dmc_tasks import dmc
from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=0)
env = RenderWrapper(env)
from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
env = FrameStackWrapper(env, 3)
env = consistent_normalize(env, normalize_obs=False)




# funtions
def vec_norm(vec):
    return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)
    
# 加载模型
path = "/data/zh/project12_Metra/METRA/exp/Quadruped/Contrastive_Phig-TB16-SZNMETRA/exp/Quadruped/Contrastive_v3-bs1024sd000_1727096456_dmc_quadruped_SZN/plots/TrajPlot_RandomZ_20.pngsd000_1727187065_dmc_quadruped_SZN"
path = path + '/'
load_option_policy_base = torch.load(path + "wandb/latest-run/filesoption_policy.pt")
load_traj_encoder_base = torch.load(path + "wandb/latest-run/filestaregt_traj_encoder.pt")
load_SampleZNetwork = torch.load(path + 'wandb/latest-run/filesSampleZPolicy.pt')
policy = load_option_policy_base['policy']
traj_encoder = load_traj_encoder_base['target_traj_encoder']
SZN = load_SampleZNetwork['goal_sample_network']

# settings：
max_path_length = 200
option_dim = load_option_policy_base['dim_option']
# path = '/data/zh/project12_Metra/METRA/tests/videos/local_test/'
Given_g = False
PhiPlot = True
LoadNpy = True


# 初始化
obs = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
Trajs = []
device = 'cuda'
num_task = 8
eval_times = 8
init_obs = torch.tensor(obs).unsqueeze(0).expand(eval_times, -1).to(device)


# # 加载goal
# support_options = torch.eye(option_dim).to(device)
if Given_g:
    all_goal_obs = []
    for i in range(num_task):
        goal_obs = env.render_goal(i)
        all_goal_obs.append(goal_obs)
    all_goal_obs_tensor = torch.tensor(all_goal_obs, dtype=torch.float)
    eval_times = num_task
    support_vec = torch.eye(option_dim).to(device)
else:
    ## discrete:
    # if option_dim == 2:
    #     directions = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
    #     support_options = torch.tensor(directions).to(device)
    # eval_times = support_options.shape[0]
    # support_options = torch.eye(option_dim).to(device)
    ## 使用SZN：
    # input_token = torch.randn_like(init_obs).to(device)
    # support_options = vec_norm(SZN(input_token).mean)
    ## 使用随机初始化
    support_options = vec_norm(torch.randn(eval_times, option_dim).to(device))

def interact_with_env():
    # interact with env
    All_Cover_list = []
    for i in trange(eval_times):
        # 初始化
        obs = env.reset()
        frames = []
        Cover_list = {}
        obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
        phi_s_0 = traj_encoder(obs_tensor).mean
        if Given_g:
            goal_tensor = torch.tile(all_goal_obs_tensor[i].reshape(-1), (3,1)).reshape(-1).unsqueeze(0).to('cuda')
            phi_g = traj_encoder(goal_tensor).mean
            # if biaoding
            # weight = (phi_g * support_vec).sum(-1)
            # index = torch.argmax(weight).cpu().numpy()
            # support_option = support_vec[index].unsqueeze(0)
            # if freeze at start 
            freeze_option = vec_norm(phi_g - phi_s_0) 
            freeze_option = vec_norm(torch.randn_like(freeze_option).to(device))
        else:
            support_option = support_options[i].unsqueeze(0)
        if PhiPlot: 
            Traj = []
            if Given_g:
                Traj.append(phi_g)
            else:
                Traj.append(support_option)
        # 每一条轨迹
        for t in trange(max_path_length):
            # policy inference:
            phi_s = traj_encoder(obs_tensor).mean
            if Given_g: 
                # to do; 需要映射；
                # option = vec_norm(phi_g - phi_s)
                option = freeze_option
            else:
                option = support_option
            obs_option = torch.cat((obs_tensor, option), -1).float()
            action_tensor = policy(obs_option)[1]['mean']
            action = action_tensor[0].detach().cpu().numpy()
            # iteration:
            obs, reward, _, info = env.step(action)
            if 'env_infos' not in Cover_list:
                Cover_list['env_infos'] = {}
                Cover_list['env_infos']['coordinates'] = []
                Cover_list['env_infos']['next_coordinates'] = []
            Cover_list['env_infos']['coordinates'].append(info['coordinates'])
            Cover_list['env_infos']['next_coordinates'].append(info['next_coordinates'])
            obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
            # for viz:
            obs_img = obs.astype(np.uint8).reshape(64,64,9)
            frames.append(obs_img[:,:,:3])
            if PhiPlot:
                Traj.append(phi_s)
                
        ## metrics:
        All_Cover_list.append(Cover_list)
        
        ## save traj. as gif
        gif_name = path+ str(i) + '.gif'
        imageio.mimsave(gif_name, frames, 'GIF', duration=20)
        # print('saved', gif_name)
        Trajs.append(Traj)
        
    eval_metrics = env.calc_eval_metrics(All_Cover_list, is_option_trajectories=True)
    print('MjNumUniqueCoords:', eval_metrics['MjNumUniqueCoords'])
    return Trajs
    
def plot_phi_traj(Trajs, load_npy_path, type='PCA'):
    if load_npy_path is not None:
        Trajs_array = np.load(load_npy_path)
    else:
        Trajs_array = torch.concatenate([torch.concatenate(i) for i in Trajs]).detach().to('cpu').numpy()    # [batch*time, dim]
        with open (path + 'Trajs.npy', 'wb') as f:
            np.save(f, Trajs_array)
    if type == 'PCA':
        # PCA to 2 dim for viz
        Trajs_array_2dim = PCA(n_components=2).fit_transform(Trajs_array)    # [batch*time, 2]
        Trajs_array_2dim = Trajs_array_2dim.reshape(eval_times, max_path_length + 1, 2)
    elif type == 'TSNE':
        tsne = TSNE(n_components=2)
        Trajs_array_2dim = tsne.fit_transform(Trajs_array)    # [batch*time, 2]
        Trajs_array_2dim = Trajs_array_2dim.reshape(eval_times, max_path_length + 1, 2)
    Trajs_goal = Trajs_array_2dim[:, 0, :]
    Trajs_array_2dim = Trajs_array_2dim[:, 1:, :]
    colors = cm.rainbow(np.linspace(0, 1, eval_times))
    plt.figure()
    for i in range(eval_times):
        plt.plot(Trajs_array_2dim[i,:,0], Trajs_array_2dim[i,:,1], color=colors[i], label=str(i))
        plt.scatter(Trajs_array_2dim[i,0,0], Trajs_array_2dim[i,0,1], color=colors[i], marker='o')
        plt.scatter(Trajs_goal[i,0], Trajs_goal[i,1], color=colors[i], marker='*', edgecolors='black')
    plt.legend()
    plt.savefig(path + 'phi_traj.png')
    print('traj plot saved', path + 'phi_traj.png')


if __name__ == '__main__':
    if LoadNpy:
        load_npy_path = path + 'Trajs.npy'
    else:
        Trajs = interact_with_env()
        load_npy_path = None
    
    if PhiPlot:
        plot_phi_traj(Trajs, load_npy_path, type='TSNE')
        
    print('done')


