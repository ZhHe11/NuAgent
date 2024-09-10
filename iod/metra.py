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


class METRA(IOD):
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
            target_theta=1,

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
        
        self.MaxLenPhi = 0
        
        # for psro:
        self.init_obs = torch.tensor(init_obs).unsqueeze(0).expand(self.num_random_trajectories, -1).to(self.device)
        self.exp_z = None   
        self.goal_sample_network = goal_sample_network.to(self.device)
        self.space_predictor = space_predictor.to(self.device)
        self.goal_sample_optim = None
        self.last_phi_g = None
        self.last_phi_g_dist = None
        self.epoch_final = None
        self.Network_None_Update_count = None
        self.sample_wait_count = 0
        self.exp_theta_dist = None
        self.UpdateSGN = 0
        self.cold_start = 1
        self.acc_buffer = torch.zeros(self.num_random_trajectories).to(self.device)
        self.acc = torch.ones(self.num_random_trajectories).to(self.device)
        self.space_predictor_optim = None
        self._trans_phi_optimization_epochs = _trans_phi_optimization_epochs
        self.target_theta = target_theta
        
    @property
    def policy(self):
        return {
            'option_policy': self.policy_for_agent,
        }
    
    def vec_norm(self, vec):
        return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)
    
    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _clip_phi_g(self, goal, lower_value=-300, upper_value=300):
        epsilon = 1e-6
        dim = goal.shape[-1]
        lower = lower_value * torch.ones(dim).to(self.device) + epsilon
        upper = upper_value * torch.ones(dim).to(self.device) + epsilon

        clip_up = (goal > upper).float()
        clip_down = (goal < lower).float()
        with torch.no_grad():
            clip = ((upper - goal) * clip_up + (lower - goal) * clip_down)

        return goal + clip

    @torch.no_grad()
    def get_return(self, traj, g=None, phi_g=None):
        if phi_g == None:
            phi_g = self.target_traj_encoder(g)
        R = 0
        discount = 0.99
        s = traj[:-1]
        s_next = traj[1:]
        phi_s = self.target_traj_encoder(s).mean
        phi_s_next = self.target_traj_encoder(s_next).mean
        # reward = (self.vec_norm(phi_s_next - phi_s) * self.vec_norm(phi_g - phi_s)).sum(dim=-1)
        # for r in reward[::-1]:
        #     R = discount * R + r
        
        # is_achieved
        distance = torch.norm(phi_g - phi_s_next[-1], dim=-1)
        distance_score =  torch.exp(1-distance)
        R += distance_score
        
        # # scale:
        # R = R / 300
        return R
    
    @torch.no_grad()
    def get_regret(self, s, a, s_next, a_next, g=None, phi_g=None, is_norm=True):
        '''
        s = [batch, seq, dim]
        a = [batch, seq, dim]
        batch_regret = [batch]
        '''
        batch_regret = []
        for i in range(s.shape[0]):
            discount = 0.99
            epoch_traj_s = s[i]
            epoch_traj_a = a[i]
            epoch_traj_s_next = s_next[i]
            epoch_traj_a_next = a_next[i]
            
            phi_s = self.target_traj_encoder(epoch_traj_s).mean
            phi_s_next = self.target_traj_encoder(epoch_traj_s_next).mean
            
            option = self.vec_norm(phi_g[i] - phi_s)
            next_option = self.vec_norm(phi_g[i] - phi_s_next)
            
            processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(epoch_traj_s), option.float())
            next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(epoch_traj_s_next), next_option.float())
            
            q = torch.min(
                self.qf1(processed_cat_obs, epoch_traj_a).flatten(),
                self.qf2(processed_cat_obs, epoch_traj_a).flatten(),
            )
            q_next = torch.min(
                self.qf1(next_processed_cat_obs, epoch_traj_a_next).flatten(),
                self.qf2(next_processed_cat_obs, epoch_traj_a_next).flatten(),
            )
            
            r = (self.vec_norm(phi_s_next - phi_s) * self.vec_norm(phi_g[i] - phi_s)).sum(dim=-1)
        
            regret = torch.abs(r + discount * q_next - q).sum()
            batch_regret.append(regret)
        
        batch_regret = torch.tensor(batch_regret).to(self.device)
        batch_regret_mean = batch_regret.mean() / self.max_path_length
        if is_norm:
            batch_regret = (batch_regret - batch_regret.mean()) / (batch_regret.std() + 1e-8)
        
        return batch_regret, batch_regret_mean
    
    
    @torch.no_grad()
    def get_R(self, phi_s_f, phi_g, sample_batch):
        train_max_count = 20
        Network_Update = torch.zeros((sample_batch)).to(self.device)
        Sample_Update = torch.zeros((sample_batch)).to(self.device)
        Network_R_std = torch.zeros((sample_batch)).to(self.device)
        if self.Network_None_Update_count == None:
            self.Network_None_Update_count = torch.zeros((sample_batch), dtype=int).to(self.device)
            self.Network_R = torch.zeros((sample_batch, train_max_count)).to(self.device)
        R = torch.zeros(sample_batch).to(self.device)
        phi_g_sf_distance_score =  1 / (1 + torch.norm(phi_g - phi_s_f, dim=-1).detach())
        # 对于每条数据独立判E定和更新；
        for i in range(sample_batch):
            R[i] = phi_g_sf_distance_score[i]
                
            self.Network_R[i][self.Network_None_Update_count[i]] = R[i]
            if self.Network_None_Update_count[i] > 0:
                if self.Network_None_Update_count[i] <= 5:
                    Network_R_std[i] = torch.std(self.Network_R[i][:self.Network_None_Update_count[i]+1])
                else:
                    Network_R_std[i] = torch.std(self.Network_R[i][self.Network_None_Update_count[i]-5 : self.Network_None_Update_count[i]+1 ])
            else:
                Network_R_std[i] = R[i]
            self.Network_None_Update_count[i] = self.Network_None_Update_count[i] + 1
                
            # 判定是否需要更新目标；
            if R[i] >= (0.1): 
                # 说明学会了，要更新网络，向更远的方向；
                Network_Update[i] = 1       # 1 是学会了
                Sample_Update[i] = 1        # 要更新phi_g
                self.Network_None_Update_count[i] = 0
                self.Network_R[i] = torch.zeros_like(self.Network_R[i])
            elif (Network_R_std[i] < 0.05 and self.Network_None_Update_count[i] > 2) or self.Network_None_Update_count[i] >= 5:
                # 说明学不会，不更新网路，但更新phi_g;
                # 我试试反向更新；我想让网络的mean有变化；
                Network_Update[i] = -1      # 是没学会；
                Sample_Update[i] = 1        # 要更新phi_g
                self.Network_None_Update_count[i] = 0
                self.Network_R[i] = torch.zeros_like(self.Network_R[i])
                
        return Network_Update, Sample_Update, R

    def AsymmetricLoss(self, value, alpha_pos=1, alpha_neg=-0.1):
        mask = torch.where(value>0, 1, 0)
        loss = alpha_pos * mask * value + alpha_neg * (1-mask) * value
        return loss 


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
            if key in ['obs', 'actions'] :
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
            # Add paths to the replay buffer
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
            
            if self.method['explore'] == 'theta' and self.epoch_final is not None:
                sample_batch = self.epoch_final['obs'].shape[0]
                final_state = self.epoch_final['obs'][:,-1]
                with torch.no_grad():
                    phi_s_0 = self.target_traj_encoder(self.init_obs).mean
                    phi_s_f = self.target_traj_encoder(final_state).mean
                s_f_theta = self.vec_norm(phi_s_f - phi_s_0)
            
                if self.goal_sample_optim is None:
                    self.goal_sample_optim = optim.SGD(self.goal_sample_network.parameters(), lr=self.dim_option * 1e-2)
                    self.last_phi_g = phi_s_f    
                    
                # 用s_f_theta更新GSN网络
                s_f_L = torch.norm(phi_s_f - phi_s_0, dim=-1) 
                theta_L = self.goal_sample_network(s_f_theta)
                theta_L_mean = self._clip_phi_g(theta_L.mean, lower_value=-self.max_path_length, upper_value=self.max_path_length)
                theta_L_stddev = theta_L.stddev
                # calculate R
                Network_Update, Sample_Update, R = self.get_R(phi_s_f=phi_s_f, phi_g=self.last_phi_g, sample_batch=sample_batch)
                # 更新网络
                # 对于mean的更新：
                loss_mean = torch.abs(Network_Update) * self.AsymmetricLoss(s_f_L - theta_L_mean.squeeze(-1),alpha_neg=-0.1, alpha_pos=1)
                # 对于分布的更新；
                loss_std = self.AsymmetricLoss(-Network_Update * 1 * theta_L_stddev.squeeze(-1), alpha_neg=1, alpha_pos=0.01)
                loss = (loss_mean + loss_std).mean()
                self.goal_sample_optim.zero_grad()
                loss.backward()
                self.goal_sample_optim.step()
                    
                # 推理，获取新的phi_g
                with torch.no_grad():
                    # random:
                    randn_exp_theta = self.vec_norm(torch.randn_like(self.last_phi_g)).to(self.device)
                    theta_L = self.goal_sample_network(randn_exp_theta)
                    theta_L_mean = self._clip_phi_g(theta_L.mean)
                    theta_L_stddev = theta_L.stddev
                        
                # 更新新的phi_g
                next_phi_g = phi_s_0 + (theta_L_mean + torch.rand_like(theta_L_stddev) * theta_L_stddev) * randn_exp_theta
                # 软更新；
                self.last_phi_g = Sample_Update.unsqueeze(-1) * next_phi_g + (1 - Sample_Update.unsqueeze(-1)) * self.last_phi_g

                np_phi_g = self.last_phi_g.detach().cpu().numpy()
                extras = self._generate_option_extras(random_options, phi_sub_goal=np_phi_g)  

                if wandb.run is not None:
                    wandb.log({
                            "theta/loss_mean": loss_mean.detach().mean(),
                            "theta/loss_std": loss_std.detach().mean(),
                            "theta/loss": loss.detach(),
                            }) 
                    Mean = theta_L_mean
                    Stddev = theta_L_stddev
                    for i in range(sample_batch):    
                        wandb.log({
                                    "theta/mean-" + str(i): float(Mean[i][0].cpu()),
                                    "theta/std-" + str(i): float(Stddev[i][0].cpu()),
                                    "theta/R-" + str(i): float(R[i].cpu()),
                                    })
            
            else: 
                extras = self._generate_option_extras(random_options)      # 变成字典的形式；
            
        return dict(
            extras=extras,
            sampler_key='option_policy',
        )
    
    '''
    Train Process;
    '''
    def _train_once_inner(self, path_data):
        # t1 = time.time()
        self._update_replay_buffer(path_data)       
        # t2 = time.time()
        # print("[_update_replay_buffer]", t2-t1) 
        epoch_data = self._flatten_data(path_data) 
        # t3 = time.time()
        # print("[_flatten_data]", t3-t2)
        tensors = self._train_components(epoch_data)  
        # t4 = time.time()
        # print("[_train_components]", t4-t3)
        return tensors
    
    '''
    Main Function;
    '''
    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}
        if self.UpdateSGN and self.cold_start == 0:
            return {}
        
        for i in range(self._trans_optimization_epochs):
            for j in range(self._trans_phi_optimization_epochs):
                # t1 = time.time()
                tensors = {}
                if self.replay_buffer is None:              
                    v = self._get_mini_tensors(epoch_data)
                else:
                    v = self._sample_replay_buffer()
                # t2 = time.time()
                # print(" [_get_mini_tensors]", t2-t1)
                self._optimize_te(tensors, v)
                # t3 = time.time()
                # print(" [_optimize_te]", t3-t2)
            # for j in range(self._trans_phi_optimization_epochs):
                # if self.replay_buffer is None:              
                #     v = self._get_mini_tensors(epoch_data)
                # else:
                #     v = self._sample_replay_buffer()
                with torch.no_grad():
                    self._update_rewards(tensors, v)
                # t4 = time.time()
                # print(" [_update_rewards]", t4-t3)
                self._optimize_op(tensors, v)
                # t5 = time.time()
                # print(" [_optimize_op]", t5-t4)
                
        return tensors

    '''
    【1】 更新phi函数；
    '''
    def _optimize_te(self, tensors, internal_vars):        
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

        self.update_target_traj(theta=self.target_theta)

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],
                )

    '''
    【2】更新qf和SAC函数；
    '''
    def _optimize_op(self, tensors, internal_vars): 
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
        )

        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars) 
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
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

    '''
    【3】更新reward；更新option；更新phi_s；
    '''
    def _update_rewards(self, tensors, v):       
        obs = v['obs']
        next_obs = v['next_obs']
        cur_z = self.traj_encoder(obs).mean
        next_z = self.traj_encoder(next_obs).mean
        if self.method['phi'] in ['soft_update', 'her_reward', 'contrastive']:   
            sub_goal = v['sub_goal']
            option = v['options']
            goal_z = self.target_traj_encoder(sub_goal).mean.detach()
            target_cur_z = self.target_traj_encoder(obs).mean.detach()
            target_next_z = self.target_traj_encoder(next_obs).mean.detach()
            option_goal = self.vec_norm(goal_z - target_cur_z)
            next_option_goal = self.vec_norm(goal_z - target_next_z)
            option_s_s_next = next_z - cur_z
            target_next_z_z = target_next_z - target_cur_z  
            ###########################################
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
                'goal_z': goal_z,
                'options': option,                   
                'option_goal': option_goal,      
                'option_s_s_next': option_s_s_next,
                'next_option_goal': next_option_goal,   
                'target_next_z_z': target_next_z_z,
            })
            ###########################################
            
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
            'PureRewardMean': rewards.mean(),           # baseline reward;
            'PureRewardStd': rewards.std(),             # baseline reward;
        })
        v['rewards'] = rewards                          # 是baseline的reward; 具体用到的reward之后再根据self.method计算；

    
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
        
        if self.method["phi"] in ['contrastive']:
            vec_phi_sample = self.target_traj_encoder(v['pos_sample']).mean
            vec_phi_s_s_next = v['option_s_s_next']
            vec_phi_sample = torch.where(torch.norm(vec_phi_sample-phi_s_next)<1e-5, v['goal_z'], vec_phi_sample)
            matrix_s_sample = vec_phi_sample.unsqueeze(0) - phi_s.unsqueeze(1)
            matrix_s_sp_norm = matrix_s_sample / (torch.norm(matrix_s_sample, p=2, dim=-1, keepdim=True) + 1e-8)
            matrix = (vec_phi_s_s_next.unsqueeze(1) * matrix_s_sp_norm).sum(dim=-1)
            inner_pos = torch.diag(matrix)     
            # 加一个判断，如果g-与g特别接近，就用mask掉；
            dist_theta = 1e-6
            distance_pos_neg = torch.norm(vec_phi_sample.unsqueeze(0) - vec_phi_sample.unsqueeze(1), p=2, dim=-1)
            mask = torch.where( distance_pos_neg < dist_theta , 0, 1)
            mask = mask + torch.eye(phi_s.shape[0], phi_s.shape[0]).to(self.device)
            matrix = matrix * mask
            # # log softmax
            # t = 1
            # matrix = matrix / t
            # label = torch.arange(matrix.shape[0]).to(self.device)
            # new_reward1 = - F.cross_entropy(matrix, label)
            # new_reward2 = - F.cross_entropy(matrix.T, label)
            # rewards = (new_reward1 + new_reward2 ) / 2
            
            
            
            
            tensors.update({
                'next_z_reward': rewards.mean(),
                'inner_s_s_next_pos': inner_pos.mean(),
                'inner_s_s_next_neg': new_reward1.mean(),
                'inner_s_s_next_neg2': new_reward2.mean(),
                'distance_pos_neg': distance_pos_neg.mean(),
            })
                 
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

            cst_penalty = cst_dist - torch.square(phi_s_next - phi_s).mean(dim=1)       
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)           
            
            if self.method["phi"] in ['contrastive']:           
                te_obj = rewards + dual_lam.detach() * cst_penalty
            else:
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
        if self.method["policy"] in ['her_reward']:
            option = v['option_goal']
            next_option = v['next_option_goal']
            goal_reward = ((v['target_next_z_z']) * option).sum(dim=1)
            policy_rewards = goal_reward * self._reward_scale_factor
            # update to logs
            tensors.update({
                'policy_rewards': policy_rewards.mean(),
                'norm_option_s_s_next': torch.norm(v['option_s_s_next'], p=2, dim=-1).mean(),
                'diff_option_g_option_sample': torch.norm((v['option_goal'] - v['options']), p=2, dim=-1).mean(),
            })
        
        else: # basline
            option = v['options']
            next_option = v['next_options']
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
            dones=v['dones'],
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
        if self.method['policy'] == "her_reward":
            option = v['option_goal'].detach()
        else:
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
        
        elif env_name == 'kitchen':
            self.eval_kitchen(runner)
            # self.eval_kitchen_metra(runner)
            
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
        env = runner._env
        fig, ax = plt.subplots()
        env.draw(ax)
        # 1. initialize the parameters
        
        max_path_length = self.max_path_length
        # goals = torch.zeros((num_eval, self.dim_option)).to(self.device)
        
        frames = []
        All_Repr_obs_list = []
        All_Goal_obs_list = []
        All_Return_list = []
        All_GtReturn_list = []
        All_trajs_list = []
        FinallDistanceList = []
        Pepr_viz = True
        np_random = np.random.default_rng()    
        
        goals_list = [
            [12.7, 16.5],
            [1.1, 12.9],
            [4.7, 4.5],
            [17.2, 0.9],
            [20.2, 20.1],
            [4.7, 0.9],
            [0.9, 4.7],
        ]
        num_eval = len(goals_list)
        goals = torch.tensor(np.array(goals_list)).to(self.device)
        
        # 2. interact with the env
        progress = tqdm(range(num_eval), desc="Evaluation")

        for i in progress:
            # 2.1 calculate the goal;
            # goal = env.env.goal_sampler(np_random)
            ax.scatter(goals_list[i][0], goals_list[i][1], s=50, marker='x', alpha=1, edgecolors='black', label='target.'+str(i))
            print(goals[i])
            # 2.2 reset the env
            obs = env.reset()  
            obs = torch.tensor(obs).unsqueeze(0).to(self.device).float()
            target_obs = env.get_target_obs(obs, goals[i])
            phi_target_obs = self.traj_encoder(target_obs).mean
            phi_obs_ = self.traj_encoder(obs).mean
            Repr_obs_list = []
            Repr_goal_list = []
            gt_return_list = []
            traj_list = {}
            traj_list["observation"] = []
            traj_list["info"] = []
            # 2.3 interact loop
            for t in range(max_path_length):
                option, phi_obs_, phi_target_obs = self.gen_z(target_obs, obs, device=self.device, ret_emb=True)
                obs_option = torch.cat((obs, option), -1).float()
                # for viz
                if Pepr_viz:
                    Repr_obs_list.append(phi_obs_.cpu().numpy()[0])
                    Repr_goal_list.append(phi_target_obs.cpu().numpy()[0])
                # get actions from policy
                # action = self.option_policy(obs_option)[1]['mean']
                action, agent_info = self.option_policy.get_action(obs_option)
                # interact with the env
                obs, reward, dones, info = env.step(action)
                gt_dist = np.linalg.norm(goals[i].cpu() - obs[:2])
                # for recording traj.2
                traj_list["observation"].append(obs)
                info['x'], info['y'] = env.env.get_xy()
                traj_list["info"].append(info)
                # calculate the repr phi
                obs = torch.tensor(obs).unsqueeze(0).to(self.device).float()
                gt_reward = - gt_dist / (30 * max_path_length)
                gt_return_list.append(gt_reward)
                
            All_Repr_obs_list.append(Repr_obs_list)
            All_Goal_obs_list.append(Repr_goal_list)
            All_GtReturn_list.append(gt_return_list)
            All_trajs_list.append(traj_list)
            FinallDistanceList.append(-gt_dist)
            progress.set_postfix_str(
                f"gt_ret={sum(gt_return_list):.3f},final_dist={gt_dist:.3f}")
            
            
        All_GtReturn_array = np.array([np.array(i).sum() for i in All_GtReturn_list])
        print(
            "All_GtReturn", All_GtReturn_array.mean()
        )
        FinallDistance = np.array(FinallDistanceList).mean()
        FinallDistSum = np.array(FinallDistanceList).sum()
        
        plot_trajectories(env, All_trajs_list, fig, ax)
        ax.legend(loc='lower right')
        
        if wandb.run is not None:
            path = wandb.run.dir
            filepath = os.path.join(path, "Maze_traj.png")
            plt.savefig(filepath) 
            print(filepath)
            wandb.log(  
                        {
                            "epoch": runner.step_itr,
                            "test/All_GtReturn": All_GtReturn_array.mean(),
                            "test/FinallDistance": FinallDistance,
                            "test/FinallDistSum": FinallDistSum,
                            "Maze_traj": wandb.Image(filepath),
                        },
                    )
        
            if Pepr_viz and self.dim_option==2:
                PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length)
                print('Repr_Space_traj saved')

                directions = self.vec_norm(torch.randn((100, self.dim_option))).to('cuda')
                dist = self.goal_sample_network(directions)
                mean = dist.mean.detach()
                stddev = dist.stddev.detach()
                edge_mean = (directions * mean).cpu().numpy()
                edge_std = (directions * (mean+stddev)).cpu().numpy()
                plt.figure(figsize=(8, 8))
                plt.scatter(x=edge_mean[:,0], y=edge_mean[:,1])
                plt.scatter(x=edge_std[:,0], y=edge_std[:,1])
                # plt.colorbar(label='Probability Density')
                plt.title('Edge')
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                img_path = os.path.join(path, "SGN.png")
                plt.savefig(img_path)

    def _save_pt(self):
        if wandb.run is not None:
            path = wandb.run.dir
        else:
            path = '.'
        file_name = path + 'option_policy.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'policy': self.option_policy,
        }, file_name)
        file_name = path + 'taregt_traj_encoder.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'target_traj_encoder': self.target_traj_encoder,
        }, file_name)
        file_name = path + 'sample_goal_network.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'goal_sample_network': self.goal_sample_network,
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
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
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
            eval_option_metrics.update({'epoch': runner.step_itr})
            wandb.log(eval_option_metrics)




        
        
        
        
        
        
        
        
        
                
