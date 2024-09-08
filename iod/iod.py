from collections import defaultdict, deque

import numpy as np
import torch

import global_context
import dowel_wrapper
from dowel import Histogram
from garage import TrajectoryBatch
from garage.misc import tensor_utils
from garage.np.algos.rl_algorithm import RLAlgorithm
from garagei import log_performance_ex
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import compute_total_norm
from iod.utils import MeasureAndAccTime
import wandb
import random


from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze, plot_trajectories, plot_value
import matplotlib.pyplot as plt
import os
from iod.ant_eval import *


class IOD(RLAlgorithm):
    def __init__(
            self,
            *,
            env_name,
            algo,
            env_spec,
            option_policy,
            traj_encoder,
            skill_dynamics,
            dist_predictor,
            dual_lam,
            optimizer,
            alpha,
            max_path_length,
            n_epochs_per_eval,
            n_epochs_per_log,
            n_epochs_per_tb,
            n_epochs_per_save,
            n_epochs_per_pt_save,
            n_epochs_per_pkl_update,
            dim_option,
            num_random_trajectories,
            num_video_repeats,
            eval_record_video,
            video_skip_frames,
            eval_plot_axis,
            name='IOD',
            device=torch.device('cpu'),
            sample_cpu=True,
            num_train_per_epoch=1,
            discount=0.99,
            sd_batch_norm=False,
            skill_dynamics_obs_dim=None,
            trans_minibatch_size=None,
            trans_optimization_epochs=None,
            discrete=False,
            unit_length=False,
            sample_type=None,
            num_her=0,
            _trans_online_sample_epochs=1,
    ):
        self.env_name = env_name
        self.algo = algo

        self.discount = discount
        self.max_path_length = max_path_length

        self.device = device
        self.sample_cpu = sample_cpu
        self.option_policy = option_policy.to(self.device)
        self.traj_encoder = traj_encoder.to(self.device)
        self.dual_lam = dual_lam.to(self.device)
        self.param_modules = {
            'traj_encoder': self.traj_encoder,
            'option_policy': self.option_policy,
            'dual_lam': self.dual_lam,
        }
        if skill_dynamics is not None:
            self.skill_dynamics = skill_dynamics.to(self.device)
            self.param_modules['skill_dynamics'] = self.skill_dynamics
        if dist_predictor is not None:
            self.dist_predictor = dist_predictor.to(self.device)
            self.param_modules['dist_predictor'] = self.dist_predictor

        self.alpha = alpha
        self.name = name

        self.dim_option = dim_option

        self._num_train_per_epoch = num_train_per_epoch
        self._env_spec = env_spec

        self.n_epochs_per_eval = n_epochs_per_eval
        self.n_epochs_per_log = n_epochs_per_log
        self.n_epochs_per_tb = n_epochs_per_tb
        self.n_epochs_per_save = n_epochs_per_save
        self.n_epochs_per_pt_save = n_epochs_per_pt_save
        self.n_epochs_per_pkl_update = n_epochs_per_pkl_update
        self.num_random_trajectories = num_random_trajectories
        self.num_video_repeats = num_video_repeats
        self.eval_record_video = eval_record_video
        self.video_skip_frames = video_skip_frames
        self.eval_plot_axis = eval_plot_axis

        assert isinstance(optimizer, OptimizerGroupWrapper)
        self._optimizer = optimizer

        self._sd_batch_norm = sd_batch_norm
        self._skill_dynamics_obs_dim = skill_dynamics_obs_dim

        if self._sd_batch_norm:
            self._sd_input_batch_norm = torch.nn.BatchNorm1d(self._skill_dynamics_obs_dim, momentum=0.01).to(self.device)
            self._sd_target_batch_norm = torch.nn.BatchNorm1d(self._skill_dynamics_obs_dim, momentum=0.01, affine=False).to(self.device)
            self._sd_input_batch_norm.eval()
            self._sd_target_batch_norm.eval()

        self._trans_minibatch_size = trans_minibatch_size
        self._trans_optimization_epochs = trans_optimization_epochs

        self.discrete = discrete
        self.unit_length = unit_length

        self.traj_encoder.eval()
        self.sample_type=sample_type
        self.num_her=num_her
        self._trans_online_sample_epochs = _trans_online_sample_epochs

    @property
    def policy(self):
        raise NotImplementedError()

    def all_parameters(self):
        for m in self.param_modules.values():
            for p in m.parameters():
                yield p

    def train_once(self, itr, paths, runner, extra_scalar_metrics={}):
        logging_enabled = ((runner.step_itr + 1) % self.n_epochs_per_log == 0)

        data = self.process_samples(paths)      # 已经输入了sample得到的数据；这里是构造了数据结构，一个进入buffer
        time_computing_metrics = [0.0]
        time_training = [0.0]

        with MeasureAndAccTime(time_training):
            tensors = self._train_once_inner(data)      # 调用了metra的函数；
                                                        # 1.更新metra中的ReplayBuffer;
                                                        # 2.把当前的path转化为tensor；
                                                        # 3.调用metra中的_train_components函数；

        performence = log_performance_ex(
            itr,
            TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            discount=self.discount,
        )
        discounted_returns = performence['discounted_returns']
        undiscounted_returns = performence['undiscounted_returns']

        train_log = {}
        if (runner.step_itr + 1) % self.n_epochs_per_log == 0:
            for k in tensors.keys():
                if tensors[k].numel() == 1:
                    train_log[k] = tensors[k].item()
                else:
                    train_log[k] = np.array2string(tensors[k].detach().cpu().numpy(), suppress_small=True)
            
        if wandb.run is not None:
            wandb.log(tensors)
                
        if wandb.run is not None:        
            wandb.log({
                        "train/step": runner.step_itr,
                        })   

        return np.mean(undiscounted_returns)
    
    '''
    核心函数：
    '''
    def train(self, runner):
        last_return = None

        with global_context.GlobalContext({'phase': 'train', 'policy': 'sampling'}):
            for _ in runner.step_epochs(
                    full_tb_epochs=0,
                    log_period=self.n_epochs_per_log,
                    tb_period=self.n_epochs_per_tb,
                    pt_save_period=self.n_epochs_per_pt_save,
                    pkl_update_period=self.n_epochs_per_pkl_update,
                    new_save_period=self.n_epochs_per_save,
            ):
                # change mode
                for p in self.policy.values():
                    p.eval()
                self.traj_encoder.eval()
                # test process
                if self.n_epochs_per_eval != 0 and runner.step_itr % self.n_epochs_per_eval == 0 and wandb.run is not None:
                    self._evaluate_policy(runner, self.env_name)
                # change mode
                for p in self.policy.values():
                    p.train()
                self.traj_encoder.train()
                # train process
                for _ in range(self._num_train_per_epoch):
                    time_sampling = [0.0]
                    with MeasureAndAccTime(time_sampling):
                        runner.step_path = self._get_train_trajectories(runner)     # 1. sample trajectories
                    last_return = self.train_once(                                  # 2. train once        
                        runner.step_itr,
                        runner.step_path,
                        runner,
                        extra_scalar_metrics={
                            'TimeSampling': time_sampling[0],
                        },
                    )
                # save model
                if runner.step_itr % 20 == 0:
                    self._save_pt()
                runner.step_itr += 1

        return last_return
    
    # 在得到option之后的具体采样
    def _get_trajectories(self,
                          runner,
                          sampler_key,
                          batch_size=None,
                          extras=None,
                          update_stats=False,
                          worker_update=None,
                          env_update=None):
        if batch_size is None:
            batch_size = len(extras)
        policy_sampler_key = sampler_key[6:] if sampler_key.startswith('local_') else sampler_key
        time_get_trajectories = [0.0]
        with MeasureAndAccTime(time_get_trajectories):
            trajectories, infos = runner.obtain_exact_trajectories(
                runner.step_itr,
                sampler_key=sampler_key,
                batch_size=batch_size,
                agent_update=self._get_policy_param_values(policy_sampler_key),     # 是option_worker中的self.agent, 使用对应的policy在线交互；
                env_update=env_update,
                worker_update=worker_update,
                extras=extras,
                update_stats=update_stats,
            )
        print(f'_get_trajectories({sampler_key}) {time_get_trajectories[0]}s')

        for traj in trajectories:
            for key in ['ori_obs', 'next_ori_obs', 'coordinates', 'next_coordinates']:
                if key not in traj['env_infos']:
                    continue
                
        '''
        plot training traj
        '''
        if (runner.step_itr + 2) % self.n_epochs_per_log == 0 and wandb.run is not None and 'phi_s' in trajectories[0]['agent_infos'].keys():
            Pepr_viz = False
            if self.env_name == 'ant_maze':
                fig, ax = plt.subplots()
                env = runner._env
                env.draw(ax)
                list_viz_traj = []
                All_Repr_obs_list = []
                All_Goal_obs_list = []
                for i in range(len(trajectories)):
                    # plot phi
                    if Pepr_viz:
                        phi_s = trajectories[i]['agent_infos']['phi_s']
                        phi_g = trajectories[i]['agent_infos']['phi_sub_goal']
                        All_Repr_obs_list.append(phi_s)
                        All_Goal_obs_list.append(phi_g)
                    
                    # plot the subgoal
                    if 'sub_goal' in trajectories[i]['agent_infos'].keys():
                        sub_goal = trajectories[i]['agent_infos']['sub_goal'][0]
                        ax.scatter(sub_goal[0], sub_goal[1], s=50, marker='x', alpha=1, edgecolors='black', label='target.'+str(i))
                    # plot the traj
                    viz_traj = {}
                    viz_traj['observation'] = trajectories[i]['observations']
                    viz_traj['info'] = []
                    for j in range(len(trajectories[i]['observations'])):
                        viz_traj['info'].append({'x':viz_traj['observation'][j][0], 'y':viz_traj['observation'][j][1]})
                    list_viz_traj.append(viz_traj)
                plot_trajectories(env, list_viz_traj, fig, ax)
                ax.legend(loc='lower right')
                path = wandb.run.dir
                filepath = os.path.join(path, "train_Maze_traj.png")
                print(filepath)
                plt.savefig(filepath) 
                wandb.log(({"train_Maze_traj": wandb.Image(filepath)}))
            # else:
            #     All_Repr_obs_list = []
            #     All_Goal_obs_list = []
            #     for i in range(len(trajectories)):
            #         # plot phi
            #         if Pepr_viz:
            #             phi_s = trajectories[i]['agent_infos']['phi_s']
            #             # phi_g = trajectories[i]['agent_infos']['phi_sub_goal']
            #             All_Repr_obs_list.append(phi_s)
            #             # All_Goal_obs_list.append(phi_g)
                        
            # if Pepr_viz:
            #     path = wandb.run.dir
            #     PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=self.max_path_length, tag='train')                
            #     print('Repr_Space_traj saved')
            
        return trajectories


    # 【训练步骤1：采样】得到训练轨迹，先得到option，然后再采样；
    def _get_train_trajectories(self, runner):
        default_kwargs = dict(
            runner=runner,
            update_stats=True,
            worker_update=dict(
                _render=False,
                _deterministic_policy=False,
            ),
            env_update=dict(_action_noise_std=None),
        )
        kwargs = dict(default_kwargs, **self._get_train_trajectories_kwargs(runner))    # 在这里设置生成options

        paths = self._get_trajectories(**kwargs)        # 在这里用计算得到的options来 sample trajectories

        return paths

    def process_samples(self, paths):   
        data = defaultdict(list)
        for i in range(len(paths)):     
            path = paths[i]
            data['obs'].append(path['observations'])
            data['next_obs'].append(path['next_observations'])
            data['actions'].append(path['actions'])
            data['rewards'].append(path['rewards'])
            data['dones'].append(path['dones'])
            data['returns'].append(tensor_utils.discount_cumsum(path['rewards'], self.discount))
            data['ori_obs'].append(path['observations'])
            data['next_ori_obs'].append(path['next_observations'])
            if 'pre_tanh_value' in path['agent_infos']:
                data['pre_tanh_values'].append(path['agent_infos']['pre_tanh_value'])
            if 'log_prob' in path['agent_infos']:
                data['log_probs'].append(path['agent_infos']['log_prob'])
            if 'option' in path['agent_infos']:
                data['options'].append(path['agent_infos']['option'])
                data['next_options'].append(np.concatenate([path['agent_infos']['option'][1:], path['agent_infos']['option'][-1:]], axis=0))
            
            traj_len = len(path['observations'])
            index = np.arange(0, traj_len)

            ## use sub_goal from path; if not exist, use the last one as subgoal;
            if 'sub_goal' in path['agent_infos']:
                data['sub_goal'].append(path["agent_infos"]["sub_goal"])
            else:
                data['sub_goal'].append(np.tile(path['observations'][-1], (traj_len, 1)))

            if 'phi_sub_goal' in path['agent_infos']:
                data['phi_sub_goal'].append(path["agent_infos"]["phi_sub_goal"])
            
            ## for contrastive positive sample：
            path_goal_dist = np.zeros(path['observations'].shape[0])
            path_subgoal = np.zeros(path['observations'].shape)
            if self.sample_type in ['contrastive']:
                data['final_goal_distance'].append(traj_len - 1 - index)
                traj_len = len(path['observations'])
                for t in range(traj_len):
                    t_pos = random.choices([_ for _ in range(0,traj_len-t)], weights=[0.9**index for index in range(0,traj_len-t)], k=1)[0]
                    t_pos = min(t_pos+2, traj_len-1-t)
                    path_goal_dist[t] = t_pos
                    path_subgoal[t] = path['observations'][t + t_pos]
                data['pos_sample_distance'].append(path_goal_dist)
                data['pos_sample'].append(path_subgoal)
    
            ## for HER resample sub_goal:
            if self.sample_type in ['her_reward']:
                num_her = self.num_her
                subgoal_indices = np.random.choice(traj_len, num_her, replace=False)
                for j in range(len(subgoal_indices)):
                    subgoal_index = subgoal_indices[j]
                    data['obs'].append(path['observations'][:subgoal_index+1])
                    data['next_obs'].append(path['next_observations'][:subgoal_index+1])
                    data['actions'].append(path['actions'][:subgoal_index+1])
                    data['rewards'].append(path['rewards'][:subgoal_index+1])
                    path['dones'][subgoal_index] = 1
                    data['dones'].append(path['dones'][:subgoal_index+1])
                    data['returns'].append(tensor_utils.discount_cumsum(path['rewards'][:subgoal_index+1], self.discount))
                    data['ori_obs'].append(path['observations'][:subgoal_index+1])
                    data['next_ori_obs'].append(path['next_observations'][:subgoal_index+1])
                    if 'pre_tanh_value' in path['agent_infos']:
                        data['pre_tanh_values'].append(path['agent_infos']['pre_tanh_value'][:subgoal_index+1])
                    if 'log_prob' in path['agent_infos']:
                        data['log_probs'].append(path['agent_infos']['log_prob'][:subgoal_index+1])
                    if 'option' in path['agent_infos']:
                        data['options'].append(path['agent_infos']['option'][:subgoal_index+1])
                        data['next_options'].append(np.concatenate([path['agent_infos']['option'][:subgoal_index+1][1:], path['agent_infos']['option'][:subgoal_index+1][-1:]], axis=0))
                    if self.sample_type in ['contrastive']:
                        data['pos_sample_distance'].append(path_goal_dist[:subgoal_index+1])
                        data['pos_sample'].append(path_subgoal[:subgoal_index+1])
                    if 'phi_sub_goal' in path['agent_infos']:
                        data['phi_sub_goal'].append(path["agent_infos"]["phi_sub_goal"][:subgoal_index+1])
                        
                    data['sub_goal'].append(np.tile(path['observations'][:subgoal_index+1][-1], (subgoal_index+1, 1)))
                
        return data


    def _get_policy_param_values(self, key):
        param_dict = self.policy[key].get_param_values()
        for k in param_dict.keys():
            if self.sample_cpu:
                param_dict[k] = {k: v.detach().cpu() for k, v in param_dict[k].items()}
            else:
                param_dict[k] = {k: v.detach() for k, v in param_dict[k].items()}
        return param_dict

    def _generate_option_extras(self, options, sub_goal=None, phi_sub_goal=None):
        extras = [{"option": option, "exploration_type": 0} for option in options]
        if sub_goal is not None:
            for i in range(len(sub_goal)):
                extras[i]["sub_goal"] = sub_goal[i]
        
        if phi_sub_goal is not None:
            for i in range(len(phi_sub_goal)):
                extras[i]["phi_sub_goal"] = phi_sub_goal[i]
                
        return extras

    def _gradient_descent(self, loss, optimizer_keys):
        self._optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        self._optimizer.step(keys=optimizer_keys)

    def _get_mini_tensors(self, epoch_data):
        num_transitions = len(epoch_data['actions'])
        idxs = np.random.choice(num_transitions, self._trans_minibatch_size)

        data = {}
        for key, value in epoch_data.items():
            data[key] = value[idxs]

        return data

    def _log_eval_metrics(self, runner):
        runner.eval_log_diagnostics()
        runner.plot_log_diagnostics()
