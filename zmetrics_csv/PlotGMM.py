import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import copy
from iod.viz_utils import *


def UpdateGMM(dists, GMM=None, mix_dist_prob=None, device='cuda'):
    if GMM is None:
        component_distribution = dist.Independent(
            dist.Normal(
                loc=torch.stack([g.mean[0] for g in dists]),
                scale=torch.stack([g.stddev[0] for g in dists])
            ),
            reinterpreted_batch_ndims=1
        )

        if mix_dist_prob is None:
            # 创建均匀的 mixture_distribution
            mixture_distribution = dist.Categorical(
                probs=(torch.ones(len(dists)) / len(dists)).to(device)
            )
        else: 
            mixture_distribution = dist.Categorical(
                probs=mix_dist_prob
            )

        # 组合成一个 MixtureSameFamily 分布
        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )

        return window_dist
    
    else:
        component_distribution = GMM.component_distribution
        mixture_distribution = mixture_distribution

        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )

        return window_dist

def PlotGMM(window_dist, psi_z, fig, ax, device, dim=4):
    x_grid = np.linspace(-1, 1, 100)
    y_grid = np.linspace(-1, 1, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points).to(device)
    zeros_tensor = torch.zeros(grid_points_tensor.shape[0], dim-2).to(device)
    grid_points_tensor_expanded = torch.cat((grid_points_tensor, zeros_tensor), dim=1)
    log_prob = window_dist.log_prob(grid_points_tensor_expanded).cpu().numpy()
    prob_density = np.exp(log_prob).reshape(X_grid.shape)
    contour = ax.contourf(X_grid, Y_grid, prob_density, levels=20, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_ticks([])
    if psi_z is not None:
        ax.scatter(psi_z[:, 0], psi_z[:, 1], alpha=0.5, color='gray', edgecolor='none', marker='o', s=5)
    # ax.set_title('GMM Probability Density')
    # ax.set_xlabel('Z[0]')
    # ax.set_ylabel('Z[1]')


## Main interation:
@torch.no_grad()
def eval_cover_rate(env, agent_traj_encoder, agent_policy, device, options=None, ax=None, max_path_length=300, Psi=None, option_type=None):
    
    All_Repr_obs_list = []
    All_Goal_obs_list = []
    All_trajs_list = []
    FinallDistanceList = []
    ArriveList = []
    All_Cover_list = []
    
    for j in trange(len(options)):
        obs_0 = env.reset()
        obs_0 = torch.tensor(obs_0).unsqueeze(0).to(device).float()
        obs = copy.deepcopy(obs_0)
        phi_obs_ = agent_traj_encoder(obs).mean
        
        Repr_obs_list = []
        Repr_goal_list = []
        traj_list = {}
        traj_list["observation"] = []
        traj_list["info"] = []
        Cover_list = {}
        option = options[j].unsqueeze(0)
        
        for t in range(max_path_length):
            phi_obs_ = agent_traj_encoder(obs).mean
            obs_option = torch.cat((obs, option), -1).float()
            Repr_obs_list.append(Psi(phi_obs_).cpu().numpy()[0])
            Repr_goal_list.append(option.cpu().numpy()[0])
            action, agent_info = agent_policy.get_action(obs_option)
            obs, reward, dones, info = env.step(action)
            traj_list["observation"].append(obs)
            
            if 'env_infos' not in Cover_list:
                Cover_list['env_infos'] = {}
                Cover_list['env_infos']['coordinates'] = []
                Cover_list['env_infos']['next_coordinates'] = []
            
            if 'coordinates' in info.keys():
                traj_list["info"].append(info['coordinates'])
                Cover_list['env_infos']['coordinates'].append(info['coordinates'])
                Cover_list['env_infos']['next_coordinates'].append(info['next_coordinates'])
            
            else:
                if hasattr(env.env, 'get_xy'):
                    info['x'], info['y'] = env.env.get_xy()
                else:
                    info['x'], info['y'] = obs[0], obs[1]
            
                traj_list["info"].append(info)
                Cover_list['env_infos']['coordinates'].append(obs[:2])
                Cover_list['env_infos']['next_coordinates'].append(obs[:2])
            
            obs = torch.tensor(obs).unsqueeze(0).to(device).float()
        
        All_Repr_obs_list.append(Repr_obs_list)
        All_Goal_obs_list.append(Repr_goal_list)
        All_trajs_list.append(traj_list)
        Cover_list['env_infos']['coordinates'] = np.array(Cover_list['env_infos']['coordinates'])
        Cover_list['env_infos']['next_coordinates'] = np.array(Cover_list['env_infos']['next_coordinates'])
        All_Cover_list.append(Cover_list)

    
    return ax, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList, All_Cover_list       
      



if __name__ == '__main__':
        
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epoch', type=int, default='0')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--eval_type', type=str, default='random')
    args = parser.parse_args()
    device = 'cuda:3'
    max_path_length = 300
    args.eval_num = 16
    args.model_path = '/mnt/nfs2/zhanghe/NuAgent/exp/Large/TheBestsd000_1735032511_ant_maze_large_SZPC'
    args.eval_type = 'random_psi'
    eval_type = args.eval_type
    args.epoch_list = [12000, 14000, 16000, 18000]
    
    for epoch in args.epoch_list:
        # 1. define the env:
        # Ant;
        from envs.mujoco.ant_env import AntEnv
        from iod.utils import get_normalizer_preset
        from garagei.envs.consistent_normalized_env import consistent_normalize

        # env = AntEnv(render_hw=100)
        # normalizer_name = 'ant'
        # normalizer_kwargs = {}
        # normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        # env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)      
        
        # AntMaze
        # from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        # args.env = 'ant_maze'
        # env = MazeWrapper("antmaze-medium-diverse-v0", random_init=False)
        args.env = 'ant_large_maze'
        from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        env = MazeWrapper("antmaze-large-diverse-v0", random_init=False)
        
        # # LM
        # from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        # args.env = 'lm'
        # env = MazeWrapper("maze2d-large-v1", random_init=False)
        
        normalizer_kwargs = {}
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
        
        obs = env.reset()
        policy_path = args.model_path + '/wandb/latest-run/filesoption_policy-' + str(epoch) + '.pt'
        traj_encoder_path = policy_path.replace('option_policy', 'traj_encoder')
        load_option_policy_base = torch.load(policy_path, map_location=device)
        load_traj_encoder_base = torch.load(traj_encoder_path, map_location=device)
        agent_policy = load_option_policy_base['policy'].eval()
        dim_option = load_traj_encoder_base['dim_option']
        agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()
        model_name = policy_path.split('/')[-4]
        path = './test/' + model_name   

        if 'psi' in eval_type:
            SZN_path =policy_path.replace('option_policy', 'SampleZPolicy')
            load_SZN_path_base = torch.load(SZN_path, map_location=device)
            SZN = load_SZN_path_base['goal_sample_network'].eval()
            input_token = load_SZN_path_base['input_token']
            qf1 = load_option_policy_base['qf1']
            qf2 = load_option_policy_base['qf2']
            alpha = load_option_policy_base['alpha']

        # 4. interaction:
        def __Psi(phi_x):
            if 'psi' in eval_type:
                return torch.tanh(2/max_path_length * (phi_x))
            else:
                return phi_x

        # Plot GMM:
        import torch.distributions as dist

        window = load_SZN_path_base['window']
        window_dist = UpdateGMM(window, device=device)
    
        psi_z=None
        fig, ax = plt.subplots(1,1)
        PlotGMM(window_dist, psi_z, fig, ax, device, dim=4)
        args.eval_num = 16
        
        options = window_dist.sample((args.eval_num, ))
        ax, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList, All_Cover_list = eval_cover_rate(env, agent_traj_encoder, agent_policy, device, options=options, ax=ax, max_path_length=300, Psi=__Psi, option_type=args.eval_type)   
        PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=False, ax=ax)
        
        
        
        
        # 美化格式
        from matplotlib import font_manager
        from pathlib import Path
        ArialPath = Path("/mnt/nfs2/zhanghe/NuAgent/fonts/Arial.ttf")
        TimesPath = font_manager.FontProperties(fname="/mnt/nfs2/zhanghe/NuAgent/fonts/Times New Roman.ttf", weight='bold')
        ax.tick_params(axis='both', labelsize=10)
        ax.set_title('Repr. Sapce', font=TimesPath, fontsize=25, pad=10)
        plt.subplots_adjust(right=0.99) 
                        
        save_path = '/mnt/nfs2/zhanghe/NuAgent/zmetrics_csv/AntLargeMaze'
        filepath = save_path + '/GMM' + str(epoch) + '.pdf'
        plt.savefig(filepath, format='pdf', bbox_inches='tight')
        print(filepath)
        
        
