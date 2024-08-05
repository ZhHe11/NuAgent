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
from iod.agent import *
from iod.ant_eval import *
from iod.update_policy import *

import torch.optim as optim



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
            
            init_obs=None,
            
            phi_type=None,
            policy_type=None,
            explore_type=None,
            
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
        add a inner reward model: RND
        '''
        # self.rnd = RNDModel(29, 1)
        self.target_traj_encoder = target_traj_encoder.to(self.device)
        self.predict_traj_encoder = predict_traj_encoder.to(self.device)

        
        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        

        '''
        wrapper for agent for online interaction.
        '''
        policy_for_agent = {
            "default_policy": self.option_policy,
            "traj_encoder": self.traj_encoder,
            # "exploration_policy": self.explore_policy,
        }
        self.policy_for_agent = AgentWrapper(policies=policy_for_agent) 
        
        self.MaxLenPhi = 0
        
        # self.exp_z = torch.randn((8, self._skill_dynamics_obs_dim), requires_grad=True).to(self.device).detach().clone().requires_grad_(True)
        # self.exp_z_optimizer = optim.Adam([self.exp_z], lr=0.1)
        # self.init_obs = torch.tensor(init_obs).unsqueeze(0).expand(8, -1).to(self.device)
        
        
    @property
    def policy(self):
        return {
            'option_policy': self.policy_for_agent,
        }
    
    def vec_norm(self, vec):
        return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)
    
    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self, runner):               # 得到options;
        if self.discrete == 1:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
        
        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
                
            if self.method['explore'] == 'buffer_explore':
                if self.replay_buffer.n_transitions_stored > 100:
                    v = self._sample_replay_buffer(batch_size=runner._train_args.batch_size)
                    buffer_subgoal = self.traj_encoder(v['sub_goal']).mean.detach().cpu().numpy()
                    buffer_state = self.traj_encoder(v['obs']).mean.detach().cpu().numpy()
                    random_options = self.vec_norm(buffer_subgoal - buffer_state)
                    
                    noise_std = 1  # 可以根据需要调整
                    noise = torch.randn(v['sub_goal'].shape) * noise_std
                    v['sub_goal'] = (v['sub_goal'].detach().cpu() + noise.float()).numpy()
                    extras = self._generate_option_extras(random_options, v['sub_goal'])  
                    
                else:
                    init_obs = self.init_obs.cpu().numpy()
                    goals_list = [
                                    [4.7, 0.9],
                                    [0.9, 4.7],
                                    [4.7, 4.5],
                                    [4.7, 0.9],
                                    [0.9, 4.7],
                                    [0, 0],
                                    [2.0, 2.0],
                                    [2.0, 0.9]
                                ]
                    goals_np = np.array(goals_list)
                    init_obs[:,:2] = goals_np 
                    extras = self._generate_option_extras(random_options, init_obs)

            elif self.method['explore'] == 'psro':
                # if self.replay_buffer.n_transitions_stored > 100:
                #     # find more difficult one;
                #     v = self._sample_replay_buffer(batch_size=runner._train_args.batch_size)
                #     self.exp_z = v['sub_goal']
                #     self.exp_z_optimizer.zero_grad()
                #     option = self.vec_norm(self.traj_encoder(self.exp_z).mean - self.traj_encoder(self.init_obs).mean)
                #     obs_option = torch.cat((self.init_obs, option), -1).float()
                #     action = self.option_policy(obs_option)[1]['mean']
                #     qf_obs_option = self._get_concat_obs(self.option_policy.process_observations(self.init_obs), option)
                #     q_value = torch.min(self.qf1(qf_obs_option, action), self.qf2(qf_obs_option, action))
                #     loss = q_value.mean()
                #     loss.backward()
                #     self.exp_z_optimizer.step()
                #     opt_subgoal = self.exp_z.detach().cpu().numpy()
                #     print("self.option", option)
                #     v['sub_goal'] = opt_subgoal
                #     extras = self._generate_option_extras(option, v['sub_goal'])
                
                pass
                
                
            elif self.method['explore'] == "freeze":
                init_obs = self.init_obs.cpu().numpy()
                goals_list = [
                                [4.7, 0.9],
                                [0.9, 4.7],
                                [4.7, 4.5],
                                [4.7, 0.9],
                                [0.9, 4.7],
                                [0, 0],
                                [2.0, 2.0],
                                [2.0, 0.9]
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

    def _sample_replay_buffer(self, batch_size=None):        # 看看是如何从buffer中加载数据的
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
    Train process
    '''
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
    def _optimize_op(self, tensors, internal_vars):         # [loss] 对于q和policy的loss
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
                
            if self.method['policy'] == 'her_reward':
                sub_goal = v['sub_goal']
                sub_goal_z = self.traj_encoder(sub_goal).mean
                option = v['options']
                goal_option = self.vec_norm(sub_goal_z - cur_z)
                v.update({
                    'sub_goal_z': sub_goal_z,
                    'goal_option': goal_option,
                })
            
            else:
                option = v['options']
            
            target_z = next_z - cur_z

            if self.discrete == 1:
                masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * option).sum(dim=1)
                rewards = inner

            # For dual objectives
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
                'options': option,
            })

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
        method_type = self.method["phi"]
        obs = v['obs']
        next_obs = v['next_obs']
        phi_s = v['cur_z']
        phi_s_next = v['next_z']

        if method_type == 'her_reward':
            '''
            zhanghe 
            我要加入goal，重新计算两个表征z_sample和z_start_goal之间的距离；
            '''
            
            # simple:
            # 这里的v['option']是采样的z，是随机分布的单位向量；
            # next_z_reward = ((v['next_z'] - v['cur_z']) * v['goal_option']).sum(dim=1)
            
            # next_z_reward = torch.minimum(next_z_reward, torch.tensor(1.0).to(self.device))
            # sub_goal_reward = (v['goal_option'] * v['options']).sum(dim=1)
            # new_reward = next_z_reward + sub_goal_reward
            
            new_reward = rewards
                              
            tensors.update({
                'next_z_reward': rewards.mean(),
                # 'sub_goal_reward': sub_goal_reward.mean(),
            })
        
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

            cst_penalty = cst_dist - torch.square(phi_s_next - phi_s).mean(dim=1)        # 这是后面的约束项，约束skill表征的大小；
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)             # 限制最大值；trick，如果惩罚项太大，会导致优化困难；
            
            if method_type == "her_reward": 
                te_obj = new_reward + dual_lam.detach() * cst_penalty   

            else :
                # 原方法：
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
    def _update_loss_qf(self, tensors, v):

        policy_type = self.method["policy"]
        
        # calculate z using obs' and obs    
        # 1. using sub_goal
        phi_obs_ = v['next_z']
        phi_obs = v['cur_z']
        
        if policy_type == "her_reward":
            '''
            zhanghe:
            change the policy learning process; using other z and reward, not the z_sample; let the train and eval the same target
            '''
            option = v['goal_option']
            next_option = self.vec_norm(v['sub_goal_z'] - v['next_z'])
            
            # arr_reward = torch.where((torch.norm(v['obs'] - v['sub_goal'], p=2, dim=-1, keepdim=True) + 1e-8)< 1e-5, 1, 0).squeeze(-1)
                
            # 对应的reward
            goal_reward = (self.vec_norm(v['next_z'] - v['cur_z']) * v['goal_option']).sum(dim=1) 
            
            # final reward
            policy_rewards = goal_reward * self._reward_scale_factor

            # update to logs
            tensors.update({
                'policy_rewards': policy_rewards.mean(),
                'goal_reward': goal_reward.mean(),
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


    def _update_loss_op(self, tensors, v):
        if self.method['policy'] == "her_reward":
            option = v['goal_option']
        else:
            option = v['options']
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
            [20.2, 20.1],
            [4.7, 0.9],
            [0.9, 4.7],
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
            target_obs = env.get_target_obs(obs, goals[i])
            phi_target_obs = self.traj_encoder(target_obs).mean
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
                option = phi_target_obs - phi_obs 
                option = option / (torch.norm(option, p=2) + 1e-8)
                print("options:", option)
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
                # skill_vec = option[:,:-1]
                option_reward = (option * delta_phi_obs).sum()
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

            
        plot_trajectories(env, All_trajs_list, fig, ax)
        ax.legend(loc='lower right')
        
        if wandb.run is not None:
            path = wandb.run.dir
            filepath = os.path.join(path, "Maze_traj.png")
            plt.savefig(filepath) 
            print(filepath)
            wandb.log(  
                        {
                            "test/Average_Return": All_Return_array.mean(),
                            "test/All_GtReturn": All_GtReturn_array.mean(),
                            "Maze_traj": wandb.Image(filepath),
                        },
                        step=runner.step_itr
                    )
        
        if Pepr_viz and self.dim_option==2:
            PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length)
            print('Repr_Space_traj saved')

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
        