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
from tqdm import trange
from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze, plot_trajectories, plot_value
from sklearn.decomposition import PCA
import matplotlib.cm as cm

import wandb
import os

import torch.nn as nn
# from iod.SAC import *
# from iod.RND import *


'''
# save the traj. as fig
有空写到test里面去，写在这个太乱了；
'''
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
        plt.scatter(All_Goal_obs_2d[start_index, 0], All_Goal_obs_2d[start_index, 1], marker='*', s=100, c=color, label="option."+str(i))
    file_path = os.path.join(path, "repr_traj.png")
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('traj. in representation space')
    plt.legend()
    plt.savefig(file_path)
    wandb.log(({"Repr_Space_traj": wandb.Image(file_path)}))


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
            
            
            predict_traj_encoder=None,
            target_traj_encoder=None,
            
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
        
        '''
        change the method here;
        this will be added into the args.parser later;
        '''
        
        self.method = {
            "eval": "no_norm",
            # "phi": "her_reward",
            "phi": "baseline",
            "policy": "sub_goal_reward", 
            # "policy": "baseline",
            "explore": "baseline",
        }
        
        '''
        add a inner reward model: RND
        '''
        # self.rnd = RNDModel(29, 1)
        self.target_traj_encoder = target_traj_encoder.to(self.device)
        self.predict_traj_encoder = predict_traj_encoder.to(self.device)
        
    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }
    
    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self, runner):               # 得到options;
        if self.discrete == 1:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
        elif self.discrete == 2:
            '''
            her + metra
            在这里定义每次sample使用的option，这个option与goal相关；
            Method 1: 依然随机生成z_sample，但是训练的时候loss控制z_s0_sn与采样用的z_sample相近；
            Method 2: 从buffer中sample一个goal，然后根据goal生成一个option；
            '''
            # Method 1
            # random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            # if self.unit_length:
            #     random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            # extras = self._generate_option_extras(random_options)
            # Method 2
            # to do ...    
            # [zhanghe] 0702
            batch_size = runner._train_args.batch_size
            mini_batch_size = int(batch_size / 2)
            # 最开始buffe为空时，先随机采样；
            if len(self.replay_buffer._buffer) == 0:
                random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
                if self.unit_length:
                    random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
                extras = self._generate_option_extras(random_options)
            # 当buffer不为空时：
            else:
                # 一部分是已有的option，【温习】已有的目标点
                # 最简单的，先从已有的buffer中提取一些obs，作为target，有点像her简单版的感觉；
                samples = self.replay_buffer.sample_transitions(mini_batch_size)
                phi_obs = self.traj_encoder(torch.tensor(samples["obs"]).to(self.device)).mean.detach()
                phi_goal = self.traj_encoder(torch.tensor(samples["goal"]).to(self.device)).mean.detach()
                goal_options = phi_goal - phi_obs
                goal_options = goal_options.cpu().numpy()
                
                # 另一部分是探索的option，【探索】从来没有探索的目标点；
                ## 最简单的方法：使用random，在找相似度最小的，但是效率肯定低；
                random_options = np.random.randn(batch_size, self.dim_option)
                if self.unit_length:
                    random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
                sim_matrix = goal_options @ random_options.T
                sum_sim = np.sum(sim_matrix, axis=0)
                sorted_indices = np.argsort(sum_sim)
                eight_smallest_indices_sort = sorted_indices[:mini_batch_size]
                random_options = random_options[eight_smallest_indices_sort]
                ## 第二种方式：用奇异值分解，找最不相似的点，QR分解、特征值分解（Eigenvalue Decomposition）、奇异值分解（Singular Value Decomposition, SVD）、Householder变换；
                ### to do

                ## 第三种方法：在高维空间，有约束option_dim > batch_size, 此时可使用Gram-Schmidt正交化，找到正交基；

                # 把两个option合在一起
                options = np.concatenate((goal_options, random_options), axis=0)
                extras = self._generate_option_extras(options)
                print("sample option for exploration and training: ", options)

        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            if self.method['explore'] == "explore-one":
                # becasue We neet explore:0719
                # 1. use two stage: (explore one, eval zero)
                ones = np.ones((random_options.shape[0], 1))
                random_options = np.concatenate([random_options, ones], axis=1)
            elif self.method['explore'] == "baseline": 
                # 2. use baseline: (all zero)
                zeros = np.zeros((random_options.shape[0], 1))
                random_options = np.concatenate([random_options, zeros], axis=1)            
            extras = self._generate_option_extras(random_options)       # 变成字典的形式；
        
        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
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

    def _sample_replay_buffer(self):        # 看看是如何从buffer中加载数据的
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data

    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)           # 这里需要修改，因为我要把subgoal加入进去；

        epoch_data = self._flatten_data(path_data)      # 本质上是，把array和list转化为tensor

        tensors = self._train_components(epoch_data)    # 训练模型，tensor是info;

        return tensors

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None:                  # 我要看他是否使用了replay buffer，使用了；
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()

            self._optimize_te(tensors, v)
            self._update_rewards(tensors, v)
            self._optimize_op(tensors, v)
            # self._optimize_op(tensors, v, ep=True)
            
        return tensors

    
    '''
    【loss1】skill表征PHI的loss:
    '''
    def _optimize_te(self, tensors, internal_vars):        
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

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
    【loss2】option policy已经SAC算法的loss:
    '''
    def _optimize_op(self, tensors, internal_vars, ep=False):         # [loss] 对于q和policy的loss
        self._update_loss_qf(tensors, internal_vars, ep)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
        )
        self._gradient_descent(
            tensors['forward_loss'],
            optimizer_keys=['predict_encoder'],
        )

        self._update_loss_op(tensors, internal_vars, ep)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)         # 这个是控制sac的entropy的；
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        sac_utils.update_targets(self)
        

    def _update_rewards(self, tensors, v):                      # 【修改】这里修改reward的计算方法；
        obs = v['obs']
        next_obs = v['next_obs']

        if self.inner:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            target_z = next_z - cur_z

            if self.discrete == 1:
                masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * v['options'][:,:-1]).sum(dim=1)
                rewards = inner

            # For dual objectives
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })
        else:
            target_dists = self.traj_encoder(next_obs)

            if self.discrete == 1:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['options'])

        tensors.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        v['rewards'] = rewards

    
    '''
    【loss1.1】skill表征PHI的loss:
    '''   
    def _update_loss_te(self, tensors, v):          # 【更新】要修改表征loss的计算方法；
        self._update_rewards(tensors, v)            # 为什么要更新reward？reward其实就是计算技能表征z与当前运动方向的内积；
        rewards = v['rewards']
        
        # 我们要用her，用subgoal得到的z作为z；
        # method_type = 'her_reward'
        method_type = self.method["phi"]
        
        
        if method_type == 'her_reward':
            '''
            zhanghe 
            我要加入goal，重新计算两个表征z_sample和z_start_goal之间的距离；
            '''
            # z_sample = v['options']
            # phi_g = self.traj_encoder(v['goal']).mean
            phi_sub_g = self.traj_encoder(v['sub_goal']).mean
            phi_s = self.traj_encoder(v['obs']).mean
            phi_s_next = self.traj_encoder(v['next_obs']).mean

            # 暂时使用表征做差：
            z_start_next = phi_s_next - phi_s
            # z_start_goal = phi_g - phi_s
            # z_next_goal = phi_g - phi_s_next
            skill_discount = 0.5
            z_train = phi_sub_g - phi_s
            # z_train_norm = z_train / (torch.norm(z_train, p=2, dim=-1, keepdim=True) + 1e-8)
            # z_start_next_norm = z_start_next / (torch.norm(z_start_next) + 1e-8)
            z_train_norm = z_train
            z_start_next_norm = z_start_next
            
            # include dist_theta
            dist_theta = 0.1
            # z_train_norm = torch.where(z_train_norm < dist_theta, 0, z_train_norm).float()
            
            
            # new_reward = (z_start_next * z_sample).sum(dim=1) + skill_discount * (z_start_goal * z_sample).sum(dim=1) 
            new_reward = (z_start_next_norm * z_train_norm.detach()).sum(dim=1)
        
            ## 【ctb】len_weight:
            # 我想让离final越近的权重越大，离final越远的权重越小；
            # 方法1： 用step作为约束；
            # len_weight = 

            # 方法2： 用与s_final的相似度作为约束；
            # min_val = 1e-2
            # max_val = 1e2
            # len_weight = 1 / (torch.norm(z_start_goal, p=2).detach() + 1e-3)
            # len_weight = torch.clamp(len_weight, min=1e-2, max=1e2)
            # norm_len_weight = (len_weight - min_val) / (max_val - min_val)
            
            tensors.update({
                'PhiReward': new_reward.mean(),
            })
        
        obs = v['obs']
        next_obs = v['next_obs']

        if self.dual_dist == 's2_from_s':           # 没有进行imagine，我觉得这个
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            tensors.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])         # 按照batch的大小，生成一个全为1的tensor；
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

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)        # 这是后面的约束项，约束skill表征的大小；
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)             # 限制最大值；trick，如果惩罚项太大，会导致优化困难；
            
            
            if method_type == "her_reward": 
                # 【ctb】增加s_final和z_sample的约束，to be finished
                # 用sub_goal得到z_train代替z_sample，使得整个训练过程和eval过程的逻辑完全一样；
                te_obj = new_reward + dual_lam.detach() * cst_penalty   

                # 【ctb】增加len_weight
                # te_obj = norm_len_weight * rewards + dual_lam.detach() * cst_penalty    
                
            else :
                # 【ori】原方法：
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

        tensors.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })
    
    '''
    【loss2】option policySAC算法的loss:
    zhanghe0716:
        policy learn:
        model: SAC
        reward: doing
    '''
    def _update_loss_qf(self, tensors, v, ep=False):

        # policy_type = "sub_goal_reward"
        # policy_type = "baseline"
        policy_type = self.method["policy"]
        if ep == True:
            policy_type = "explore"
        
        if policy_type == "sub_goal_reward":
            '''
            zhanghe:
            change the policy learning process; using other z and reward, not the z_sample; let the train and eval the same target
            '''
            # calculate z using obs' and obs
            # 1. using final goal
            # phi_goal = self.traj_encoder(v['goal']).mean.detach()
            # phi_obs_ = self.traj_encoder(v['next_obs']).mean.detach()
            # phi_obs = self.traj_encoder(v['obs']).mean.detach()
            # option = phi_goal - phi_obs
            # next_option = phi_goal - phi_obs_        
            
            # 2. using sub_goal
            phi_goal = self.traj_encoder(v['sub_goal']).mean.detach()
            phi_obs_ = self.traj_encoder(v['next_obs']).mean.detach()
            phi_obs = self.traj_encoder(v['obs']).mean.detach()
            option = phi_goal - phi_obs
            next_option = phi_goal - phi_obs_ 
            # distance_option = 1 / (option + 1e-2) # 这个不行
            distance_next_option = torch.norm(next_option, p=2, dim=-1, keepdim=True)
            distance_option = torch.norm(option, p=2, dim=-1, keepdim=True)
            
            # distance_reward = - distance_next_option
            # relative distance
            w1 = 10
            w2 = 0.1
            dist_theta = 0.01
            
            delta_phi_norm = (phi_obs_ - phi_obs) / (torch.norm(phi_obs_ - phi_obs, p=2, dim=-1, keepdim=True) + 1e-8)
            option_norm = option / (distance_option + 1e-8)
            option_norm = torch.where(distance_option < dist_theta, 0, option_norm).float()
            # 相对的reward
            relative_dist_reward = distance_option - distance_next_option
            
            # distance reward: 
            dist_reward = 10 * (dist_theta - distance_next_option.squeeze(-1))
            dist_reward = torch.where(dist_reward > 0, 1, 0).float()
            
            # RND: exploration reward
            predict_next_feature = self.predict_traj_encoder(v["next_obs"]).mean
            target_next_feature = self.target_traj_encoder(v["next_obs"]).mean.detach()
            exp_reward = 400 * ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).detach()
            forward_mse = nn.MSELoss(reduction='none')
            update_proportion = 0.25
            forward_loss = forward_mse(predict_next_feature, target_next_feature).mean(-1)
            mask = torch.rand(len(forward_loss)).to(self.device)
            mask = (mask < update_proportion).type(torch.FloatTensor).to(self.device)
            forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
            
            # final reward: 
            goal_reward = (delta_phi_norm * option_norm).sum(dim=1) + dist_reward + exp_reward + relative_dist_reward
            # goal_reward = ((phi_obs_ - phi_obs) * norm_option).sum(dim=1) + (distance_option - distance_next_option)
            policy_rewards = goal_reward * self._reward_scale_factor

            # ground truth reward:
            distance_xy = torch.norm(v['obs'][:2] - v['sub_goal'][:2], p=2, dim=-1, keepdim=True)
            
            # update to logs
            tensors.update({
                'policy_rewards': policy_rewards.mean(),
                'inner_reward': (delta_phi_norm * option_norm).sum(dim=1).mean(),
                'relative_dist_reward': relative_dist_reward.mean(),
                'dist_reward': dist_reward.mean(),
                'exp_reward': exp_reward.mean(),
                'xy_dsitance_g_s': distance_xy.mean(),
                'forward_loss': forward_loss.mean(),
            })
        elif policy_type == "explore": 
            option = v['options']
            next_option = v['next_options']
            policy_rewards = v['rewards'] * self._reward_scale_factor
            tensors.update({
                'policy_rewards': policy_rewards.mean(),
            })
            
        else: # basline
            option = v['options']
            next_option = v['next_options']
            policy_rewards = v['rewards'] * self._reward_scale_factor
            tensors.update({
                'policy_rewards': policy_rewards.mean(),
            })
            
        if ep == False:    
            if option.shape[1] < v['options'].shape[1]:
                zeros = torch.zeros([option.shape[0], 1], dtype=float).to(self.device)
                option = torch.cat([option, zeros], dim=1)
                next_option = torch.cat([next_option, zeros], dim=1)
            else:
                option[:,-1] = 0
                next_option[:,-1] = 0
        else:
            option[:,-1] = 1
            next_option[:,-1] = 1
            
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), option.float())
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), next_option.float())
        
        # goal_reward = - torch.norm(next_option, p=2, dim=-1)
        # wandb.log(({"goal_reward": goal_reward.mean()}))
        
        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],   
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=policy_rewards,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })


    def _update_loss_op(self, tensors, v, ep=False):
        option = v['options']
        if ep == False:    
            # zeros = torch.zeros([option.shape[0], 1], dtype=float).to(self.device)
            # option = torch.cat([option, zeros], dim=1)
            option[:,-1] = 0
        else:
            # ones = torch.ones([option.shape[0], 1], dtype=float).to(self.device)
            # option = torch.cat([option, ones], dim=1)           
            option[:,-1] = 1
            
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), option)
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )


    '''
    Evaluation
    '''
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
        
        Pepr_viz = True
        np_random = np.random.default_rng()    
        
        goals_list = [
            [12.7, 16.5],
            [1.1, 12.9],
            [4.7, 4.5],
            [17.2, 0.9],
            [20.2, 20.1]
        ]
        num_eval = len(goals_list)
        goals = torch.tensor(np.array(goals_list)).to(self.device)
        
        # 2. interact with the env
        for i in trange(num_eval):
            # 2.1 calculate the goal;
            # goal = env.env.goal_sampler(np_random)
            ax.scatter(goals_list[i][0], goals_list[i][1], s=50, marker='x', alpha=1, edgecolors='black', label='target.'+str(i))
            print(goals[i])
            # 2.2 reset the env
            obs = env.reset()  
            obs = torch.tensor(obs).unsqueeze(0).to(self.device).float()
            phi_obs_ = self.traj_encoder(obs).mean
            Repr_obs_list = []
            Repr_goal_list = []
            option_return_list = []
            gt_return_list = []
            traj_list = {}
            traj_list["observation"] = []
            traj_list["info"] = []
            # 2.3 interact loop
            for t in range(max_path_length):
                # calculate the phi_obs
                phi_obs = phi_obs_
                # calculate the option:
                target_obs = env.get_target_obs(obs, goals[i])
                phi_target_obs = self.traj_encoder(target_obs).mean
                option = phi_target_obs - phi_obs 
                if self.method["eval"] == "norm": 
                    option = option / torch.norm(option, p=2)   
                # explore or not 
                ep = False
                if ep == False:    
                    zeros = torch.zeros([option.shape[0], 1], dtype=float).to(self.device)
                    option = torch.cat([option, zeros], dim=1)
                else:
                    ones = torch.ones([option.shape[0], 1], dtype=float).to(self.device)
                    option = torch.cat([option, ones], dim=1)   
                obs_option = torch.cat((obs, option), -1).float()
                # for viz
                if Pepr_viz:
                    Repr_obs_list.append(phi_obs.cpu().detach().numpy()[0])
                    Repr_goal_list.append(phi_target_obs.cpu().detach().numpy()[0])

                # get actions from policy

                action = self.option_policy(obs_option)[1]['mean']

                # interact with the env
                obs, reward, done, info = env.step(action.cpu().detach().numpy()[0])
                gt_dist = np.linalg.norm(goals[i].cpu() - obs[:2])
                
                # for recording traj.2
                traj_list["observation"].append(obs)
                info['x'], info['y'] = env.env.get_xy()
                traj_list["info"].append(info)
                
                # calculate the repr phi
                obs = torch.tensor(obs).unsqueeze(0).to(self.device).float()
                phi_obs_ = self.traj_encoder(obs).mean
                delta_phi_obs = phi_obs_ - phi_obs
                
                # option_reward and return
                skill_vec = option[:,:-1]
                option_reward = (skill_vec * delta_phi_obs).sum()
                option_return_list.append(option_reward.cpu().detach().numpy())
                gt_reward = - gt_dist / (30 * max_path_length)
                gt_return_list.append(gt_reward)
                
            All_Repr_obs_list.append(Repr_obs_list)
            All_Goal_obs_list.append(Repr_goal_list)
            All_Return_list.append(option_return_list)
            All_GtReturn_list.append(gt_return_list)
            All_trajs_list.append(traj_list)
            
        All_Return_array = np.array([np.array(i).sum() for i in All_Return_list])
        All_GtReturn_array = np.array([np.array(i).sum() for i in All_GtReturn_list])
        print(
            "Average_Return:", All_Return_array.mean(), '\n',
            "All_GtReturn", All_GtReturn_array.mean()
        )
        wandb.log(  
                    {
                        "test/Average_Return": All_Return_array.mean(),
                        "test/All_GtReturn": All_GtReturn_array.mean()
                    },
                    step=runner.step_itr
                )
        
        plot_trajectories(env, All_trajs_list, fig, ax)
        ax.legend(loc='lower right')
        path = wandb.run.dir
        filepath = os.path.join(path, "Maze_traj.png")
        print(filepath)
        plt.savefig(filepath) 
        wandb.log(({"Maze_traj": wandb.Image(filepath)}))
        
        if Pepr_viz:
            PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length)
            print('Repr_Space_traj saved')

        # save model
        # torch.save(self.traj_encoder.state_dict(), 'traj_encoder.pth')
        # torch.save(self.option_policy.state_dict(), 'option_policy.pth')

        # file_name = os.path.join('option_policy.pt')
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