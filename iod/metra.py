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
# from iod.SAC import *
# from iod.RND import *
from iod.agent import *
from iod.ant_eval import *
from iod.update_policy import *

import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal


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
            
            phi_type=None,
            policy_type=None,
            explore_type=None,
            
            goal_sample_network=None,

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
        self.init_obs = torch.tensor(init_obs).unsqueeze(0).expand(8, -1).to(self.device)
        self.exp_z = None   
        self.goal_sample_network = goal_sample_network.to(self.device)
        self.goal_sample_optim = None
        self.last_phi_g = None
        self.last_phi_g_dist = None
        self.epoch_final = None
        self.Network_None_Update_count = None
        
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
    def get_regret(self, s, a, s_next, a_next, g=None, phi_g=None):
        discount = 0.99
        s = s
        phi_s = self.target_traj_encoder(s).mean
        phi_s_next = self.target_traj_encoder(s_next).mean
        
        option = phi_g - phi_s
        next_option = phi_g - phi_s_next
        
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(s).unsqueeze(0), option.float().unsqueeze(0))
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(s_next).unsqueeze(0), next_option.float().unsqueeze(0))
        
        q = torch.min(
            self.qf1(processed_cat_obs, a.unsqueeze(0)).flatten(),
            self.qf2(processed_cat_obs, a.unsqueeze(0)).flatten(),
        )
        q_next = torch.min(
            self.qf1(next_processed_cat_obs, a_next.unsqueeze(0)).flatten(),
            self.qf2(next_processed_cat_obs, a_next.unsqueeze(0)).flatten(),
        )
        
        r = (self.vec_norm(phi_s_next - phi_s) * self.vec_norm(phi_g - phi_s)).sum(dim=-1)
    
        regret = torch.abs(r + discount * q_next - q)
        return regret
    
    @torch.no_grad()
    def get_R(self, phi_s_f, phi_g, sample_batch):
        train_max_count = 100
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
            if R[i] >= 0.5: 
                # 说明学会了，要更新网络，向更远的方向；
                Network_Update[i] = 1       # 1 是学会了
                Sample_Update[i] = 1        # 要更新phi_g
                self.Network_None_Update_count[i] = 0
                self.Network_R[i] = torch.zeros_like(self.Network_R[i])
            elif (Network_R_std[i] < 0.05 and self.Network_None_Update_count[i] > 10) or self.Network_None_Update_count[i] >= 50:
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
        num_sample_batch = 8
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
            if key == 'obs':
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
                
            if self.method['explore'] == 'buffer_explore':
                if self.replay_buffer.n_transitions_stored > 100:
                    v = self._sample_replay_buffer(batch_size=runner._train_args.batch_size)
                    buffer_subgoal = self.traj_encoder(v['sub_goal']).mean.detach()
                    buffer_state = self.traj_encoder(v['obs']).mean.detach()
                    random_options = self.vec_norm(buffer_subgoal - buffer_state).cpu().numpy()
                    
                    noise_std = 1  # 可以根据需要调整
                    noise = torch.randn(v['sub_goal'].shape) * noise_std
                    v['sub_goal'] = (v['sub_goal'].detach().cpu() + noise.float()).numpy()
                    extras = self._generate_option_extras(random_options, v['sub_goal'])  
                    
                else:
                    extras = self._generate_option_extras(random_options)  

            elif self.method['explore'] == 'psro':
                # 暂时不用，先固定goal得到一定的效果；
                if self.epoch_final is not None:
                    ## 4. using psro
                    # update network
                    s_0 = self.init_obs
                    phi_s_0 = self.target_traj_encoder(s_0).mean.detach()
                    sample_batch = self.epoch_final['obs'].shape[0]
                    final_state = self.epoch_final['obs'][:,-1]
                    self.last_final_state = final_state
                    np_phi_g = np.zeros((sample_batch, self.dim_option))
                    with torch.no_grad():
                        phi_s_f = self.target_traj_encoder(final_state).mean
                    # initial
                    if self.last_phi_g is None:
                        self.last_phi_g = torch.zeros((sample_batch, self.dim_option)).to(self.device)
                        self.goal_sample_optim = optim.SGD(self.goal_sample_network.parameters(), lr=1e-3)
                        dist = self.goal_sample_network(phi_s_0)
                        phi_g = self._clip_phi_g(dist.rsample())
                        phi_g_log_probs = dist.log_prob(phi_g)
                        self.last_phi_g = phi_g
                        self.last_phi_g_log_probs = phi_g_log_probs
                            
                    else: 
                        train_max_count = 100
                        Network_Update = torch.zeros((sample_batch)).to(self.device)
                        Sample_Update = torch.zeros((sample_batch)).to(self.device)
                        Network_R_std = torch.zeros((sample_batch)).to(self.device)
                        if self.Network_None_Update_count == None:
                            self.Network_None_Update_count = torch.zeros((sample_batch), dtype=int).to(self.device)
                            self.Network_R = torch.zeros((sample_batch, train_max_count)).to(self.device)
                        R = torch.zeros(sample_batch).to(self.device)
                        regret = torch.zeros(sample_batch).to(self.device)
                        phi_g = self.last_phi_g
                        phi_g_sf_distance_score =  1 / (1 + torch.norm(phi_g - phi_s_f, dim=-1).detach())
                        
                        # 对于每条数据独立判E定和更新；
                        for i in range(sample_batch):
                            # 计算R；
                            # to do;
                            # R[i] = self.get_return(traj=self.epoch_final['obs'][i], phi_g=self.last_phi_g[i])
                            R[i] = phi_g_sf_distance_score[i]
                            
                            # regret[i] = self.get_regret(self.epoch_final['obs'][i][-2], 
                            #                          self.epoch_final['actions'][i][-2], 
                            #                          self.epoch_final['obs'][i][-1], 
                            #                          self.epoch_final['actions'][i][-1], 
                            #                          g=None, 
                            #                          phi_g=self.last_phi_g[i])
                            
                            self.Network_R[i][self.Network_None_Update_count[i]] = R[i]
                            if self.Network_None_Update_count[i] > 0:
                                Network_R_std[i] = torch.std(self.Network_R[i][:self.Network_None_Update_count[i]+1])
                            else:
                                Network_R_std[i] = R[i]
                            self.Network_None_Update_count[i] = self.Network_None_Update_count[i] + 1
                            
                            # 判定是否需要更新目标；
                            if R[i] >= 1: 
                                # 说明学会了，要更新网络，向更远的方向；
                                Network_Update[i] = -1
                                Sample_Update[i] = 1
                                self.Network_None_Update_count[i] = 0
                                self.Network_R[i] = torch.zeros_like(self.Network_R[i])
                            elif (Network_R_std[i] < 0.05 and self.Network_None_Update_count[i] > 2) or self.Network_None_Update_count[i] >= 10:
                                # 说明学不会，不更新网路，但更新phi_g;
                                # 我试试反向更新；我想让网络的mean有变化；
                                Network_Update[i] = 1
                                Sample_Update[i] = 1
                                self.Network_None_Update_count[i] = 0
                                self.Network_R[i] = torch.zeros_like(self.Network_R[i])
                                
                                
                            # Network_Update[i] = 1
                        
                        # 更新网络
                        dist = self.goal_sample_network(phi_s_0)
                        phi_g_log_probs = dist.log_prob(self.last_phi_g.detach())
                        loss = (Network_Update * (phi_g_log_probs * R)).mean()
                        self.goal_sample_optim.zero_grad()
                        loss.backward()
                        self.goal_sample_optim.step()

                        # 推理，获取新的phi_g
                        dist = self.goal_sample_network(phi_s_0)
                        phi_g_next_i = self._clip_phi_g(dist.rsample())
                        # 更新新的phi_g
                        self.last_phi_g = Sample_Update.unsqueeze(-1) * phi_g_next_i + (1 - Sample_Update.unsqueeze(-1)) * self.last_phi_g
                                
                        for i in range(sample_batch):
                            print(
                                f"phi_g_next_i={str(self.last_phi_g[i].detach().cpu().numpy()).ljust(20)} "
                                f"DScore={str(phi_g_sf_distance_score[i].detach().cpu().numpy()).ljust(20)} "
                                f"R={str(R[i].cpu().numpy()).ljust(20)} "
                                f"R_std={str(Network_R_std[i].cpu().numpy()).ljust(20)}"
                                f"Network_Update={str(Network_Update[i].cpu().numpy()).ljust(1)}"
                            )

                            if wandb.run is not None:
                                wandb.log({
                                            # "sample/distance_score" + str(i): phi_g_sf_distance_score[i],
                                            "sample1/R" + str(i): R[i],
                                            # "sample2/regret" + str(i): regret[i],
                                           }) 
                        
                        Mean = dist.mean.detach()[0]
                        Stddev = dist.stddev.detach()[0]
                        print(
                            f"DistMean={str(Mean).ljust(20)} "
                            f"DistStddev={str(Stddev).ljust(20)} "
                        )
                        if wandb.run is not None:
                            wandb.log({
                                    "SampleDist/loss": loss.detach(),
                                    "SampleDist/phi_g_log_probs": phi_g_log_probs.mean(),
                                    }) 
                            for i in range(self.dim_option):    
                                wandb.log({
                                            "SampleDist/mean-" + str(i): float(Mean[i].cpu()),
                                            "SampleDist/std-" + str(i): float(Stddev[i].cpu()),
                                            }) 
                    np_phi_g = self.last_phi_g.detach().cpu().numpy()
                    extras = self._generate_option_extras(random_options, phi_sub_goal=np_phi_g)  

                else:
                    extras = self._generate_option_extras(random_options)  
               
            elif self.method['explore'] == 're-sample':
                if self.epoch_final is None:
                    extras = self._generate_option_extras(random_options)  
                else:
                    sample_batch = self.epoch_final['obs'].shape[0]
                    final_state = self.epoch_final['obs'][:,-1]
                    self.last_final_state = final_state
                    with torch.no_grad():
                        phi_s_f = self.target_traj_encoder(final_state).mean
                        dist = self.target_traj_encoder(final_state)
                        phi_s_next = self._clip_phi_g(dist.rsample())
                    np_phi_g = phi_s_next.detach().cpu().numpy()
                    extras = self._generate_option_extras(random_options, phi_sub_goal=np_phi_g)  
            
            elif self.method['explore'] == 'random-direction':
                if self.epoch_final is None:
                    self.phi_knew = torch.zeros((600, 600))
                if self.epoch_final is not None:
                    epoch_final_obs = self.epoch_final['obs']
                    for i in range(len(epoch_final_obs)):
                        phi_s = self.target_traj_encoder(epoch_final_obs[i]).mean
                        phi_s_position = phi_s.int().cpu() + 300
                        self.phi_knew[phi_s_position[:,0], phi_s_position[:,1]] = 1
                        
                    y_list, x_list = torch.where(self.phi_knew == 1)
                    plt.figure()
                    plt.scatter(x=x_list,y=y_list)
                    plt.savefig('img.png')
                    plt.close()
                
                phi_g_next = 300 * random_options
                extras = self._generate_option_extras(random_options, phi_sub_goal=phi_g_next)  
                
            elif self.method['explore'] == 'theta':
                if self.epoch_final is not None:
                    sample_batch = self.epoch_final['obs'].shape[0]
                    final_state = self.epoch_final['obs'][:,-1]
                    with torch.no_grad():
                        phi_s_0 = self.target_traj_encoder(self.init_obs).mean
                        phi_s_f = self.target_traj_encoder(final_state).mean
                    
                    exp_theta = self.vec_norm(phi_s_f - phi_s_0)
            
                    if self.goal_sample_optim is None:
                        self.goal_sample_optim = optim.SGD(self.goal_sample_network.parameters(), lr=1e-2)
                        self.last_phi_g = phi_s_f
                        
                    s_f_L = torch.norm(phi_s_f - phi_s_0, dim=-1) 
                    theta_L = self.goal_sample_network(exp_theta)
                    theta_L_mean = self._clip_phi_g(theta_L.mean, lower_value=-300, upper_value=300)
                    theta_L_stddev = theta_L.stddev
                    s_g_L = torch.norm(self.last_phi_g - phi_s_0, dim=-1) 
                    s_g_L_log_probs = theta_L.log_prob(s_g_L)
                    # calculate R
                    Network_Update, Sample_Update, R = self.get_R(phi_s_f=phi_s_f, phi_g=self.last_phi_g, sample_batch=sample_batch)
                    # 更新网络
                    # 对于mean的更新：
                    loss_mean = torch.abs(Network_Update) * self.AsymmetricLoss(s_f_L - theta_L_mean.squeeze(-1),alpha_neg=-0.1, alpha_pos=1)
                    # normal_dist = Normal(theta_L_mean.squeeze(-1), theta_L_stddev.squeeze(-1))
                    # s_f_L_cdf = normal_dist.cdf(s_f_L)
                    # loss_mean = 100 * torch.clip(s_f_L_cdf, max=0.4)
                    # 对于分布的更新；
                    loss_std = self.AsymmetricLoss(-Network_Update * 1 * theta_L_stddev.squeeze(-1), alpha_neg=1, alpha_pos=0.01)
                    # K = 10
                    # loss_std = torch.abs((theta_L_stddev.squeeze(-1) * (torch.norm(self.last_phi_g - phi_s_f, dim=-1) + 1e-5) - K ))
                    loss = (loss_mean + loss_std).mean()
                    self.goal_sample_optim.zero_grad()
                    loss.backward()
                    self.goal_sample_optim.step()
                    
                    # 推理，获取新的phi_g
                    with torch.no_grad():
                        # 1) freeze:
                        # freeze_direction = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
                        # freeze_direction = torch.tensor(freeze_direction, dtype=torch.float32).to(self.device)
                        # randn_exp_theta = self.vec_norm(freeze_direction)
                        # 2) random:
                        randn_exp_theta = self.vec_norm(torch.randn_like(self.last_phi_g)).to(self.device)
                        theta_L = self.goal_sample_network(randn_exp_theta)
                        theta_L_mean = self._clip_phi_g(theta_L.mean)
                        theta_L_stddev = theta_L.stddev
                        # 3) 启发式:
                        # num_sample = 10
                        # randn_exps = torch.randn((num_sample, 2)).to(self.device)
                        # randn_exps = self.vec_norm(randn_exps)
                        # theta_Ls = self.goal_sample_network(randn_exps)
                        # theta_Ls_mean = self._clip_phi_g(theta_Ls.mean, lower_value=-300, upper_value=300)
                        # theta_Ls_stddev = theta_Ls.stddev
                        # values, indices = torch.topk(theta_Ls_stddev.squeeze(-1), sample_batch)
                        # theta_L_mean = theta_Ls_mean[indices]
                        # theta_L_stddev = theta_Ls_stddev[indices]
                        # randn_exp_theta = randn_exps[indices]
        
                    # 更新新的phi_g
                    next_phi_g = phi_s_0 + (theta_L_mean + torch.rand_like(theta_L_stddev) * theta_L_stddev) * randn_exp_theta
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
                        if runner.step_itr % 50 == 0:
                            path = wandb.run.dir
                            file_name = os.path.join(path, 'SampleGoalNet.pt')
                            torch.save({
                                'SampleGoalNet': self.goal_sample_network,
                            }, file_name)
                else:
                    extras = self._generate_option_extras(random_options)  
            

            elif self.method['explore'] == "freeze":
                init_obs = self.init_obs.cpu().numpy()
                goals_list = [
                                [12.7, 16.5],
                                [1.1, 12.9],
                                [4.7, 4.5],
                                [17.2, 0.9],
                                [20.2, 20.1],
                                [4.7, 0.9],
                                [0.9, 4.7],
                                [5.0, 5.0],
                            ]
                goals_np = np.array(goals_list)
                init_obs[:,:2] = goals_np
                extras = self._generate_option_extras(random_options, init_obs)
            
            elif self.method['explore'] == "baseline": 
                extras = self._generate_option_extras(random_options)      # 变成字典的形式；
            
        return dict(
            extras=extras,
            sampler_key='option_policy',
        )
    
    '''
    Train Process;
    '''
    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)           # 这里需要修改，因为我要把subgoal加入进去；
        epoch_data = self._flatten_data(path_data)      # 本质上是，把array和list转化为tensor
        tensors = self._train_components(epoch_data)    # 训练模型，tensor是info;
        return tensors
    
    '''
    Main Function;
    '''
    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}
            if self.replay_buffer is None:              
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()
                
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
        )
        if self.method['phi'] == 'baseline':
            self.update_target_traj(theta=1)
        else:
            self.update_target_traj(theta=2e-5)

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
    def _update_rewards(self, tensors, v):                      # 【修改】这里修改reward的计算方法；
        obs = v['obs']
        next_obs = v['next_obs']
        cur_z = self.traj_encoder(obs).mean
        next_z = self.traj_encoder(next_obs).mean               # 试试不detach

        if self.method['phi'] in ['soft_update', 'her_reward', 'contrastive']:   
            sub_goal = v['sub_goal']
            option = v['options']
            # 最终方向和采样方向加权；
            goal_z = 0.5 * (self.target_traj_encoder(sub_goal).mean.detach() + v['phi_sub_goal'])
            # goal_z = v['phi_sub_goal']
            final_goal_z = v['phi_sub_goal']
            
            target_cur_z = self.target_traj_encoder(obs).mean.detach()
            target_next_z = self.target_traj_encoder(next_obs).mean.detach()
            option_goal = self.vec_norm(goal_z - target_cur_z)
            next_option_goal = self.vec_norm(goal_z - target_next_z)
            
            option_final_goal = self.vec_norm(final_goal_z - target_cur_z)
            
            option_s_s_next = next_z - cur_z
            ###########################################
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
                'goal_z': goal_z,
                'options': option,                      # 是online采样时候用的option；
                'option_goal': option_goal,             # 是phi_sub_g - phi_s / norm()
                'option_s_s_next': option_s_s_next,     # 是phi_s_next - phi_s
                'next_option_goal': next_option_goal,   # 为了输入next_option
                'final_goal_z': final_goal_z,
                # 'option_final_goal': option_final_goal,           # 采样traj.的目标goal。为了标定；
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
        option_s_s_next = v['option_s_s_next']
        
        if self.method["phi"] in ['contrastive']:
            option_goal = v['option_goal']
            # option_goal_detach = v['option_goal'].detach()
            samples = self.traj_encoder(v['pos_sample']).mean
            
            # zhanghe:0819
            # 为了标定，使用:final_goal_z
            # final_goal_z = v['final_goal_z']
            # final_g_weight = 0.1
            # samples = (1-final_g_weight) * samples + final_g_weight * final_goal_z
            
            # discount weight
            discount = 0.99
            w = discount ** (v['pos_sample_distance'])
            vec_phi_s_s_next = phi_s_next - phi_s
            vec_phi_sample = samples
            
            matrix_s_sample = vec_phi_sample.unsqueeze(0) - phi_s.unsqueeze(1)
            matrix_s_sp_norm = matrix_s_sample / (torch.norm(matrix_s_sample, p=2, dim=-1, keepdim=True) + 1e-8)
            matrix = (vec_phi_s_s_next.unsqueeze(1) * matrix_s_sp_norm).sum(dim=-1)
                        
            # 加一个判断，如果g-与g特别接近，就用mask掉；
            dist_theta = 1e-3
            distance_pos_neg = torch.norm(vec_phi_sample.unsqueeze(0) - vec_phi_sample.unsqueeze(1), p=2, dim=-1)
            mask = torch.where( distance_pos_neg < dist_theta , 0, 1)
            matrix = matrix * mask
                               
            mask_pos = torch.eye(phi_s.shape[0], phi_s.shape[0]).to(self.device)
            inner_pos = torch.diag(matrix)
            inner_neg = (matrix * (1 - mask_pos)).sum(dim=1) / (phi_s.shape[0]-1)
            new_reward = w * torch.log(F.sigmoid(inner_pos)) + w * torch.log(1 - F.sigmoid((inner_neg - 0.25)))
            
            rewards = new_reward
            tensors.update({
                'next_z_reward': rewards.mean(),
                'inner_s_s_next_pos': inner_pos.mean(),
                'inner_s_s_next_neg': inner_neg.mean(),
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

            cst_penalty = cst_dist - torch.square(phi_s_next - phi_s).mean(dim=1)        # 这是后面的约束项，约束skill表征的大小；
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)             # 限制最大值；trick，如果惩罚项太大，会导致优化困难；
            
            if self.method["phi"] in ['contrastive']:
                te_obj = rewards + dual_lam.detach() * cst_penalty    
            else:
                te_obj = rewards + dual_lam.detach() * cst_penalty                      # 这是最终的loss： reward + 惩罚项；

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
            
            # arr_reward = torch.where((torch.norm(v['obs'] - v['sub_goal'], p=2, dim=-1, keepdim=True) + 1e-8)< 1e-5, 1, 0).squeeze(-1)
                
            # 对应的reward
            assert v['option_s_s_next'].shape == option.shape, (v['option_s_s_next'].shape, option.shape)
            # goal_reward = ((v['option_s_s_next']) * option).sum(dim=1) 
            # 引入标定:final_goal_z
            goal_reward = ((v['option_s_s_next']) * option).sum(dim=1) 
            
            # final reward
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
    def _evaluate_policy(self, runner):
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

            directions = self.vec_norm(torch.randn((100, 2))).to('cuda')
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

        file_name = 'option_policy.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'policy': self.option_policy,
        }, file_name)
        file_name = 'traj_encoder.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'traj_encoder': self.traj_encoder,
        }, file_name)
        
        
        
        