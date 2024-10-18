import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians

import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze, plot_trajectories, plot_value
from sklearn.decomposition import PCA
import matplotlib.cm as cm

import wandb
import os

import torch.nn as nn
from iod.agent import *
from iod.ant_eval import *
from iod.update_policy import *

import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal

import time
from iod.GradCLipper import GradClipper

from iod.BufferDataset import BufferDataset
from torch.utils.data import DataLoader
from scipy.stats import multivariate_normal

def calc_eval_metrics(trajectories, is_option_trajectories, coord_dims=[0,1]):
    eval_metrics = {}
    coords = []
    for traj in trajectories:
        traj1 = traj['env_infos']['coordinates'][:, coord_dims]
        traj2 = traj['env_infos']['next_coordinates'][-1:, coord_dims]
        coords.append(traj1)
        coords.append(traj2)
    coords = np.concatenate(coords, axis=0)
    uniq_coords = np.unique(np.floor(coords), axis=0)
    eval_metrics.update({
        'MjNumUniqueCoords': len(uniq_coords),
    })
    return eval_metrics

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

def viz_SZN_dist(SZN, input_token, path):
    dist = SZN(input_token)
    # Data
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x,y)
    from scipy.stats import multivariate_normal
    num = dist.mean.shape[0]
    fig = plt.figure(figsize=(18, 12), facecolor='w')
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
        ax = fig.add_subplot(2, num//2, i+1, projection='3d')
        ax.plot_surface(X, Y, pd, cmap='viridis', linewidth=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')
        ax.set_title(label = str(mu_x)[:3] + '-' + str(sigma_x)[:3] + '\n' + str(mu_y)[:3] + '-' + str(sigma_y)[:3])
    plt.savefig(path + '-all' + '.png')
    plt.close()


class SZN_PPAU(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,

            replay_buffer,
            min_buffer_size,
            inner,
            num_alt_samples,
            split_group,

            dual_reg,
            dual_slack,
            dual_dist,

            pixel_shape=None,
            
            init_obs=None,
            
            phi_type="baseline",
            policy_type="baseline",
            explore_type="baseline",
            
            goal_sample_network=None,
            space_predictor=None,
            _trans_phi_optimization_epochs=1,
            _trans_policy_optimization_epochs=1,
            target_theta=1,

            SampleZNetwork=None,
            SampleZPolicy=None,
            
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape

        assert self._trans_optimization_epochs is not None
        
        self.method = {
            "eval": "norm",
            "phi": phi_type,
            "policy": policy_type,
            "explore": explore_type,
        }
        
        '''
        for updating option;
        '''
        self.target_traj_encoder = copy.deepcopy(self.traj_encoder)
        
        '''
        wrapper for agent for online interaction.
        '''
        policy_for_agent = {
            "default_policy": self.option_policy,
            "traj_encoder": self.traj_encoder,
            "target_traj_encoder": self.target_traj_encoder,
        }
        self.policy_for_agent = AgentWrapper(policies=policy_for_agent) 
        
        # for psro:
        self.init_obs = torch.tensor(init_obs).unsqueeze(0).expand(self.num_random_trajectories, -1).to(self.device)
        self.exp_z = None   
        self.epoch_final = None
        self._trans_phi_optimization_epochs = _trans_phi_optimization_epochs
        self._trans_policy_optimization_epochs = _trans_policy_optimization_epochs
        self.target_theta = target_theta
        
        self.last_z = None
        self.SampleZPolicy = SampleZPolicy.to(self.device)
        self.SampleZPolicy_optim = optim.Adam(self.SampleZPolicy.parameters(), lr=1e-4)
        self.grad_clip = GradClipper(clip_type='clip_norm', threshold=3, norm_type=2)
        
        self.input_token = torch.eye(self.num_random_trajectories).float().to(self.device)
        
        self.last_policy = copy.deepcopy(self.option_policy)
        self.last_qf1 = copy.deepcopy(self.qf1)
        self.last_qf2 = copy.deepcopy(self.qf2)
        self.last_alpha = copy.deepcopy(self.log_alpha)
        self.copyed = 0
        
    
    @property
    def policy(self):
        return {
            'option_policy': self.policy_for_agent,
        }
    
    def vec_norm(self, vec):
        return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)
    
    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    '''
    For soft-update
    '''
    def update_target_traj(self, theta=2e-5):
        for t_param, param in zip(self.target_traj_encoder.parameters(), self.traj_encoder.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - theta) + param.data * theta)
    
    def _flatten_data(self, data):
        epoch_data = {}
        epoch_final = {}
        num_her = self.num_her
        num_sample_batch = self.num_random_trajectories
        for key, value in data.items():
            if key in ['dones']:
                dones = np.concatenate(value, axis=0)
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
            # for explore_type != baseline
            if key in ['obs', 'actions', 'options'] :
                traj_key_dim = value[0].shape[-1]
                epoch_key_final = torch.zeros((num_sample_batch, self.max_path_length, traj_key_dim), dtype=torch.float32, device=self.device)
                for i in range(num_sample_batch):
                    traj_shape = value[(num_her+1) * i].shape
                    epoch_key_final[i][:traj_shape[0]] = torch.tensor(value[(num_her+1) * i], dtype=torch.float32, device=self.device)
                    if traj_shape[0] < self.max_path_length:
                        epoch_key_final[i][traj_shape[0]:] = torch.tensor(value[(num_her+1) * i][-1], dtype=torch.float32, device=self.device)
                epoch_final[key] = epoch_key_final
        
        self.epoch_final = epoch_final
        return epoch_data
    

    def _update_replay_buffer(self, data):
        if self.replay_buffer is not None:
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self, batch_size=None): 
        if batch_size == None:
            batch_size = self._trans_minibatch_size
        samples = self.replay_buffer.sample_transitions(batch_size)
        data = {}
        for key, value in samples.items():
            if key in ['rewards', 'returns', 'ori_obs', 'next_ori_obs', 'pre_tanh_values', 'log_probs']:
                continue
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)

        return data


    @torch.no_grad()
    def EstimateValue(self, policy, alpha, qf1, qf2, option, state, num_samples=10):
        '''
        num_samles越大,方差越小,偏差不会更小;
        '''
        batch = option.shape[0]
        # [s0, z]
        processed_cat_obs = self._get_concat_obs(policy.process_observations(state), option.float())                            # [b,dim_s+dim_z]
        
        # dist of pi(a|[s0, z])
        dist, info = policy(processed_cat_obs)    # [b, dim]
        actions = dist.sample((num_samples,))          # [n, b, dim]
        log_probs = dist.log_prob(actions).squeeze(-1)  # [n, b]
        
        processed_cat_obs_flatten = processed_cat_obs.repeat(1, num_samples).view(batch * num_samples, -1)      # [n*b, dim_s+z]
        actions_flatten = actions.view(batch * num_samples, -1)     # [n*b, dim_a]
        q_values = torch.min(qf1(processed_cat_obs_flatten, actions_flatten), qf2(processed_cat_obs_flatten, actions_flatten))      # [n*b, dim_1]
        
        alpha = alpha.param.exp()
            
        values = q_values - alpha * log_probs.view(batch*num_samples, -1)      # [n*b, 1]
        values = values.view(num_samples, batch, -1)        # [n, b, 1]
        E_V = values.mean(dim=0)        # [b, 1]

        return E_V.squeeze(-1)
    
    
    '''
    【0】 计算online时的option；
    '''
    def _get_train_trajectories_kwargs(self, runner):
        if self.discrete == 1:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
        
        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            
            if self.method['explore'] == 'SZN' and self.epoch_final is not None:
                if runner.step_itr % 50 == 0:
                    use_Regret = 1
                    for t in range(50):
                        dist_z = self.SampleZPolicy(self.input_token)
                        z = dist_z.sample() 
                        z_logp = dist_z.log_prob(z)
                        V_z = self.EstimateValue(policy=self.option_policy, alpha=self.log_alpha, qf1=self.qf1, qf2=self.qf2, option=z, state=self.init_obs)
                        
                        if self.copyed and use_Regret:
                            V_z_last_iter = self.EstimateValue(policy=self.last_policy, alpha=self.last_alpha, qf1=self.last_qf1, qf2=self.last_qf2, option=z, state=self.init_obs)
                        else:
                            V_z_last_iter = 0
                            
                        if use_Regret:
                            Regret = V_z - V_z_last_iter
                            V_szn = Regret
                        else:
                            V_z = (V_z - V_z.mean()) / (V_z + 1e-3)
                            V_szn = -V_z
                        
                        self.SampleZPolicy_optim.zero_grad()    
                        w = 0.001
                        loss_SZP = (-z_logp * V_szn - w * dist_z.entropy()).mean()
                        loss_SZP.backward()
                        self.grad_clip.apply(self.SampleZPolicy.parameters())
                        self.SampleZPolicy_optim.step()
                        if wandb.run is not None:
                            wandb.log({
                                "SZN/loss_SZP": loss_SZP,
                                "SZN/logp": z_logp.mean(),
                                "SZN/V_z": V_z.mean(),
                                "SZN/entropy": dist_z.entropy().mean(),
                                "epoch": runner.step_itr,
                            })
                            
                    # save k-1 policy and qf
                    def copy_params(ori_model, target_model):
                        for t_param, param in zip(target_model.parameters(), ori_model.parameters()):
                            t_param.data.copy_(param.data)
                    copy_params(self.option_policy, self.last_policy)
                    copy_params(self.log_alpha, self.last_alpha)
                    copy_params(self.qf1, self.last_qf1)
                    copy_params(self.qf2, self.last_qf2)
                    self.copyed = 1
                    
                
                    
                psi_g = self.SampleZPolicy(self.input_token).sample().detach()
                self.last_z = psi_g
                
                np_z = self.last_z.cpu().numpy()
                print("Sample Z: ", np_z)
                extras = self._generate_option_extras(np_z, psi_g=psi_g.cpu().numpy())                
            
            
            elif self.method['explore'] == 'uniform' and self.epoch_final is not None:
                # w/o unit_length
                random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
                print(random_options)
                extras = self._generate_option_extras(random_options, psi_g=random_options)
            
            else: 
                self.last_z = torch.tensor(random_options, dtype=torch.float32).to(self.device)
                extras = self._generate_option_extras(random_options, psi_g=random_options)      # 变成字典的形式；
            
        return dict(
            extras=extras,
            sampler_key='option_policy',
        )
    
    '''
    Train Process;
    '''
    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)       
        epoch_data = self._flatten_data(path_data)
        tensors = self._train_components(epoch_data)  
        return tensors
    
    '''
    Main Function;
    '''
    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}
        tensors = {}
        dataset = BufferDataset(self.replay_buffer._buffer, len=self.replay_buffer.n_transitions_stored)
        dataloader = DataLoader(dataset, batch_size=self._trans_minibatch_size, shuffle=True, num_workers=2, multiprocessing_context='fork')
        
        for epoch_i, v in enumerate(dataloader):
            if epoch_i > self._trans_optimization_epochs:
                break
            v = {key: value.type(torch.float32).to(self.device) for key, value in v.items()}
            self._optimize_te(tensors, v)
            with torch.no_grad():
                self._update_rewards(tensors, v)
            self._optimize_op(tensors, v)   
            
        return tensors

    '''
    【1】 更新phi函数；
    '''
    def _optimize_te(self, tensors, internal_vars):        
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
            params=self.traj_encoder.parameters(),
        )

        self.update_target_traj(theta=self.target_theta)

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors['LossDualLam'],
                optimizer_keys=['dual_lam'],
                params=[self.dual_lam.param],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],
                    params=self.dist_predictor.parameters(),
                )

    '''
    【2】更新qf和SAC函数；
    '''
    def _optimize_op(self, tensors, internal_vars): 
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
            params=list(self.qf1.parameters()) + list(self.qf2.parameters()),
        )

        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
            params=self.option_policy.parameters(),
        )

        self._update_loss_alpha(tensors, internal_vars) 
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
            params=[self.log_alpha.param],
        )

        sac_utils.update_targets(self)
    
    @torch.no_grad()
    def gen_z(self, sub_goal, obs, device="cpu", ret_emb: bool = False):
        traj_encoder = self.target_traj_encoder.to(device)
        goal_z = traj_encoder(sub_goal).mean
        target_cur_z = traj_encoder(obs).mean

        z = self.vec_norm(goal_z - target_cur_z)
        if ret_emb:
            return z, target_cur_z, goal_z
        else:
            return z
        
        
    @torch.no_grad()
    def gen_psi_z(self, sub_goal, obs, obs_0, device="cpu", ret_emb: bool = False):
        traj_encoder = self.target_traj_encoder.to(device)
        goal_z = traj_encoder(sub_goal).mean
        target_cur_z = traj_encoder(obs).mean
        z_0 = traj_encoder(obs_0).mean
        
        z = self.Psi(goal_z) - self.Psi(z_0)
        
        if ret_emb:
            return z, target_cur_z, goal_z
        else:
            return z
        
    
    def Psi(self, phi_x):
        return torch.tanh(1/300 * phi_x)
    
    def norm(self, x, keepdim=False):
        return torch.norm(x, p=2, dim=-1, keepdim=keepdim)        

    '''
    【3】更新reward；更新option；更新phi_s；
    '''
    def _update_rewards(self, tensors, v):       
        obs = v['obs']
        next_obs = v['next_obs']
        cur_z = self.traj_encoder(obs).mean
        next_z = self.traj_encoder(next_obs).mean
        
        if self.method["phi"] in ['Projection']:
            psi_g = v['options']
            # phi_s_0 = self.traj_encoder(v['s_0']).mean
            phi_s = cur_z
            phi_s_next = next_z
            # psi_g = v['psi_g']
            
            # psi_s_0 = self.Psi(phi_s_0)
            psi_s = self.Psi(phi_s)
            psi_s_next = self.Psi(phi_s_next)
            grad_psi_s = (1 - self.Psi(phi_s)**2).detach()
            # 0. updated option
            updated_option = psi_g
            updated_next_option = psi_g
            k = 10
            d = 1 / self.max_path_length
            
            
            # 1. Similarity Reward
            delta_norm = self.norm((psi_s_next - psi_s))
            # direction_sim = (self.vec_norm(psi_s_next - psi_s) * self.vec_norm(psi_g - psi_s)).sum(dim=-1)
            # phi_obj = 1/d * torch.clamp(delta_norm, min=-1*d, max=1*d) * direction_sim
            # reward_sim = self.max_path_length * ((psi_s_next - psi_s) * self.vec_norm(psi_g)).sum(dim=-1)
            phi_obj = 1/d * torch.clamp(delta_norm, min=-1*d, max=1*d) * (self.vec_norm(psi_s) * self.vec_norm(psi_g)).sum(dim=-1)

            # 2. Goal Arrival Reward
            reward_g_distance = 1/d * torch.clamp(self.norm(psi_g - psi_s) - self.norm(psi_g - psi_s_next), min=-k*d, max=k*d)
            reward_g_arrival = torch.where(self.norm(psi_g - psi_s_next)<d, 1.0, 0.).to(self.device)
            reward_g_dir = 1 * (self.vec_norm(psi_s_next - psi_s) * self.vec_norm(psi_g-psi_s)).sum(dim=-1)
            policy_rewards = 1 * reward_g_distance + 1 * reward_g_dir
            
            # 3. Constraints
            ## later in phi loss: cst_penalty
            # rewards = reward_sim + policy_rewards
            
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
                'rewards': policy_rewards,
                'policy_rewards': policy_rewards,
                'psi_s': psi_s,
                'psi_s_next': psi_s_next,
                # 'psi_s_0': psi_s_0,
                'updated_option': updated_option,
                "updated_next_option": updated_next_option,
                'grad_psi_s': grad_psi_s,
            })
            tensors.update({
                # 'PureRewardMean': rewards.mean(),  
                # 'PureRewardStd': rewards.std(),  
                'phi_obj': phi_obj.mean(),
                'reward_g_distance': reward_g_distance.mean(),
                'reward_g_arrival': reward_g_arrival.mean(),
                'reward_g_dir': reward_g_dir.mean(),
            })
            
            return

        else: 
            option_s_s_next = next_z - cur_z
            option = v['options']
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
                'options': option,
                'option_s_s_next': option_s_s_next,
            })

        # 如果z是one-hot形式：
        if self.discrete == 1:
            masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
            rewards = (option_s_s_next * masks).sum(dim=1)
        else:
            inner = (option_s_s_next * option).sum(dim=1)
            rewards = inner
            
        tensors.update({
            'PureRewardMean': rewards.mean(),  
            'PureRewardStd': rewards.std(),  
        })
        v['rewards'] = rewards                  
        v['policy_rewards'] = rewards

    
    '''
    【1.1】计算phi函数的loss
    '''
    def compute_loss(self):
        raise NotImplementedError
   
    def _update_loss_te(self, tensors, v): 
        self._update_rewards(tensors, v)      
        rewards = v['rewards']
        obs = v['obs']
        next_obs = v['next_obs']
        phi_s = v['cur_z']
        phi_s_next = v['next_z']

        if self.dual_dist == 's2_from_s':    
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            tensors.update({
                'LossDp': loss_dp,
            })
        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])   
            elif self.dual_dist == 's2_from_s':
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                tensors.update({
                    'ScalingFactor': scaling_factor.mean(dim=0),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
                })
            else:
                raise NotImplementedError
            
            cst_penalty_2 = 1 - self.max_path_length * (self.norm(v['psi_s']-v['psi_s_next']))
            # cst_penalty_3 = - self.norm(v['psi_s_0'])
                        
            cst_penalty = torch.clamp(cst_penalty_2, max=self.dual_slack)
                        
            # te_obj = rewards + torch.clamp(dual_lam.detach(), max=100) * (cst_penalty) + torch.clamp(cst_penalty_3, max=self.dual_slack)
            # te_obj = rewards + torch.clamp(dual_lam.detach(), max=100) * (cst_penalty)
            te_obj = rewards + dual_lam.detach() * cst_penalty
                    
            v.update({
                'cst_penalty': cst_penalty
            })
            tensors.update({
                'DualCstPenalty': cst_penalty.mean(),
            })
        else:
            te_obj = rewards
        loss_te = -te_obj.mean()
        tensors.update(
            {
                "TeObjMean": te_obj.mean(),
                "LossTe": loss_te,
                "Norm(phi_s)": self.norm(phi_s).mean(),
            }
        )
    '''
    【1.2】更新dual_lam；正则项的权重；
    '''
    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })
    
    '''
    【2.1】计算qf的reward
    '''
    def _update_loss_qf(self, tensors, v):
        if "updated_option" in v.keys():    
            option = v['updated_option']
            next_option = v['updated_next_option']
        else:
            option = v['options']   
            next_option = v['next_options']

        if 'policy_rewards' in v.keys():    
            policy_rewards = v['policy_rewards'] * self._reward_scale_factor
        else:
            policy_rewards = v['rewards'] * self._reward_scale_factor
            
        tensors.update({
            'policy_rewards': policy_rewards.mean(),
        })
    
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), option.float())
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), next_option.float())
        
        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],   
            next_obs=next_processed_cat_obs,
            dones=v['dones'].squeeze(-1),
            rewards=policy_rewards,
            policy=self.option_policy,
            qf1=self.qf1,
            qf2=self.qf2,
            alpha=self.log_alpha,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            loss_type='',
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    '''
    【2.2】计算policy的loss；
    '''
    def _update_loss_op(self, tensors, v):
        option = v['options'].detach()
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), option)
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
            qf1=self.qf1,
            qf2=self.qf2,
            alpha=self.log_alpha,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            loss_type='',   
        )
        
    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v, alpha=self.log_alpha, loss_type='', 
        )


    '''
    Evaluation
    '''
    @torch.no_grad()
    def _evaluate_policy(self, runner, env_name):
        if env_name == 'ant_maze':  
            self.eval_maze(runner)
            
        else:
            self.eval_metra(runner)
            
    def eval_kitchen(self, runner):
        import imageio
        # 初始化
        env = runner._env
        
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
            obs = env.reset()
            frames = []
            obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to('cuda')
            goal_tensor = torch.tile(all_goal_obs_tensor[i].reshape(-1), (3,1)).reshape(-1).unsqueeze(0).to('cuda')
            phi_g = self.target_traj_encoder(goal_tensor).mean
            
            for t in trange(self.max_path_length):
                # policy
                phi_s = self.target_traj_encoder(obs_tensor).mean
                option = self.vec_norm(phi_g - phi_s)
                print('option:', option)
                obs_option = torch.cat((obs_tensor, option), -1).float()
                action_tensor = self.option_policy(obs_option)[1]['mean']
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
            
            filepath = wandb.run.dir
            gif_name = filepath + str(i) + '.gif'
            imageio.mimsave(gif_name, frames, 'GIF', duration=1)
            print('saved', gif_name)
            
        print('metric_success_task_relevant:', metric_success_task_relevant)
        print('metric_success_all_objects:', metric_success_all_objects)
        if wandb.run is not None:
            wandb.log({
                'metric_success_task_relevant': sum(metric_success_task_relevant.values()) / len(metric_success_task_relevant),
                'metric_success_all_objects': sum(metric_success_all_objects.values()) / len(metric_success_all_objects),
                'epoch': runner.step_itr,
            })
    
    def eval_maze(self, runner):
        '''
        this is for zero-shot task evaluation;
        right now in ant_maze env;
        later will move to other envs(ketchen or ExORL or gyms);
        '''
        num_eval = 5
        env = runner._env
        fig, ax = plt.subplots()
        env.draw(ax)
        # 1. initialize the parameters
        max_path_length = self.max_path_length
        
        frames = []
        All_Repr_obs_list = []
        All_Goal_obs_list = []
        All_Return_list = []
        All_GtReturn_list = []
        All_trajs_list = []
        FinallDistanceList = []
        Pepr_viz = True
        np_random = np.random.default_rng()    
        
        # 2. interact with the env
        GoalList = [
            [12.7, 16.5],
            [1.1, 12.9],
            [4.7, 4.5],
            [17.2, 0.9],
            [20.2, 20.1],
            [4.7, 0.9],
            [0.9, 4.7],
        ]
        # GoalList = env.env.goal_sampler(np_random, freq=1)
        num_eval = len(GoalList)
        options = np.random.randn(num_eval, self.dim_option)
        All_Cover_list = []
        progress = tqdm(range(num_eval), desc="Evaluation")
        for i in progress:
            obs = env.reset()
            option = torch.tensor(options[i]).unsqueeze(0).to(self.device)
            obs = torch.tensor(obs).unsqueeze(0).to(self.device).float()
            phi_obs_ = self.traj_encoder(obs).mean
            phi_obs0 = copy.deepcopy(phi_obs_)
            # goal condition
            goal = GoalList[i]
            ax.scatter(goal[0], goal[1], s=25, marker='o', alpha=1, edgecolors='black')
            tensor_goal = torch.tensor(goal).to('cuda')
            obs_goal = copy.deepcopy(obs)
            obs_goal = env.get_target_obs(obs_goal, tensor_goal)
            phi_g = self.traj_encoder(obs_goal).mean
            # option
            option = self.Psi(phi_g) - self.Psi(phi_obs0)
            
            Repr_obs_list = []
            Repr_goal_list = []
            traj_list = {}
            traj_list["observation"] = []
            traj_list["info"] = []
            Cover_list = {}
            for t in range(max_path_length):
                phi_obs_ = self.traj_encoder(obs).mean
                obs_option = torch.cat((obs, option), -1).float()
                psi_obs = self.Psi(phi_obs_)
                
                # for viz
                Repr_obs_list.append(psi_obs.cpu().numpy()[0])
                Repr_goal_list.append(self.Psi(phi_g).cpu().numpy()[0])
                # get actions from policy
                action, agent_info = self.option_policy.get_action(obs_option)
                # interact with the env
                obs, reward, dones, info = env.step(action)
                # for recording traj.2
                traj_list["observation"].append(obs)
                # info['x'], info['y'] = env.env.get_xy()
                info['x'], info['y'] = obs[0], obs[1]
                traj_list["info"].append(info)
                # calculate the repr phi
                if 'env_infos' not in Cover_list:
                    Cover_list['env_infos'] = {}
                    Cover_list['env_infos']['coordinates'] = []
                    Cover_list['env_infos']['next_coordinates'] = []
                Cover_list['env_infos']['coordinates'].append(obs[:2])
                Cover_list['env_infos']['next_coordinates'].append(obs[:2])
                
                obs = torch.tensor(obs).unsqueeze(0).to(self.device).float()
                
            All_Repr_obs_list.append(Repr_obs_list)
            All_Goal_obs_list.append(Repr_goal_list)
            All_trajs_list.append(traj_list)
            Cover_list['env_infos']['coordinates'] = np.array(Cover_list['env_infos']['coordinates'])
            Cover_list['env_infos']['next_coordinates'] = np.array(Cover_list['env_infos']['next_coordinates'])
            All_Cover_list.append(Cover_list)
        
        
        eval_metrics = calc_eval_metrics(All_Cover_list, is_option_trajectories=True)
        print(eval_metrics)
        plot_trajectories(env, All_trajs_list, fig, ax)
        ax.legend(loc='lower right')

        if wandb.run is not None:
            path = wandb.run.dir + '/E' + str(runner.step_itr) + '-'
            plt.savefig(path + 'Maze_traj.png') 
            wandb.log(  
                        {
                            "epoch": runner.step_itr,
                            "SampleSteps": runner.step_itr * self.max_path_length * self.num_random_trajectories,
                            "CoordsCover": eval_metrics['MjNumUniqueCoords'], 
                            "Maze_traj": wandb.Image(path + 'Maze_traj.png'),
                        },
                    )
        
            if Pepr_viz and self.dim_option==2:
                PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=self.max_path_length, is_goal=True)
                viz_SZN_dist(self.SampleZPolicy, self.input_token, path=path)


    def _save_pt(self, epoch):
        if wandb.run is not None:
            path = wandb.run.dir
        else:
            path = '.'
        file_name = path + 'option_policy-' + str(epoch) + '.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'policy': self.option_policy,
        }, file_name)
        file_name = path + 'taregt_traj_encoder-' + str(epoch) + '.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'target_traj_encoder': self.target_traj_encoder,
        }, file_name)
        file_name = path + 'SampleZPolicy-' + str(epoch) + '.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'input_token': self.input_token,
            'goal_sample_network': self.SampleZPolicy,
        }, file_name)
        
    def eval_kitchen_metra(self, runner):
        if self.discrete == 1:
            random_options = np.eye(self.dim_option)
        else:
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
            random_options = self.vec_norm(random_options)
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=True,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )
        eval_option_metrics = {}
        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        
        record_video(runner, 'Video_RandomZ', random_trajectories, skip_frames=self.video_skip_frames)
        
        if wandb.run is not None:
            eval_option_metrics.update({'epoch': runner.step_itr})
            wandb.log(eval_option_metrics)
             
    def eval_metra(self, runner):
        num_eval_traj = 8
        
        if self.discrete:
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):
                    random_options.append(eye_options[i])
                    colors.append(i)
            random_options = np.array(random_options)
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            # if self.method['explore'] == 'SZN':
            #     random_options = self.vec_norm(self.SampleZPolicy(self.input_token[:num_eval_traj]).sample().detach()).cpu().numpy()
            # else: 
            random_options = np.random.randn(num_eval_traj, self.dim_option)
            if self.unit_length:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
            random_option_colors = get_option_colors(random_options * 4)
        
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        option_dists = self.traj_encoder(last_obs)

        option_means = option_dists.mean.detach().cpu().numpy()
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

        option_colors = random_option_colors

        with FigManager(runner, f'PhiPlot') as fm:
            draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
            draw_2d_gaussians(
                option_samples,
                [[0.03, 0.03]] * len(option_samples),
                option_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        eval_option_metrics = {}

        # Videos
        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    if self.method['explore'] == 'SZN':
                        video_options = self.vec_norm(self.SampleZPolicy(self.input_token[:9]).sample().detach()).cpu().numpy()
                    else: 
                        video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(video_options),
                worker_update=dict(
                    _render=True,
                    _deterministic_policy=True,
                ),
            )
            record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        if wandb.run is not None:
            eval_option_metrics.update({
                'epoch': runner.step_itr,
                'Steps': runner.step_itr * self.num_random_trajectories * self.max_path_length,
                })
            wandb.log(eval_option_metrics)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
                
