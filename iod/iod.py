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


from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze, plot_trajectories, plot_value
import matplotlib.pyplot as plt
import os


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
            # 'predict_encoder': 
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

    @property
    def policy(self):
        raise NotImplementedError()

    def all_parameters(self):
        for m in self.param_modules.values():
            for p in m.parameters():
                yield p

    def train_once(self, itr, paths, runner, extra_scalar_metrics={}):
        logging_enabled = ((runner.step_itr + 1) % self.n_epochs_per_log == 0)

        data = self.process_samples(paths)      # 已经输入了sample得到的数据；这里是构造了数据结构，一个进入buffer的训练元组；

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
                wandb.log(tensors, step=runner.step_itr)

        # if logging_enabled:
        #     prefix_tabular = global_context.get_metric_prefix()
        #     with dowel_wrapper.get_tabular().prefix(prefix_tabular + self.name + '/'), dowel_wrapper.get_tabular(
        #             'plot').prefix(prefix_tabular + self.name + '/'):
        #         def _record_scalar(key, val):
        #             dowel_wrapper.get_tabular().record(key, val)

        #         def _record_histogram(key, val):
        #             dowel_wrapper.get_tabular('plot').record(key, Histogram(val))

        #         for k in tensors.keys():
        #             if tensors[k].numel() == 1:
        #                 _record_scalar(f'{k}', tensors[k].item())
        #             else:
        #                 _record_scalar(f'{k}', np.array2string(tensors[k].detach().cpu().numpy(), suppress_small=True))
                
                # with torch.no_grad():
                #     total_norm = compute_total_norm(self.all_parameters())
                #     _record_scalar('TotalGradNormAll', total_norm.item())
                #     for key, module in self.param_modules.items():
                #         total_norm = compute_total_norm(module.parameters())
                #         _record_scalar(f'TotalGradNorm{key.replace("_", " ").title().replace(" ", "")}', total_norm.item())
                # for k, v in extra_scalar_metrics.items():
                    # _record_scalar(k, v)
                # _record_scalar('TimeComputingMetrics', time_computing_metrics[0])
                # _record_scalar('TimeTraining', time_training[0])

                # path_lengths = [
                #     len(path['actions'])
                #     for path in paths
                # ]
                # _record_scalar('PathLengthMean', np.mean(path_lengths))
                # _record_scalar('PathLengthMax', np.max(path_lengths))
                # _record_scalar('PathLengthMin', np.min(path_lengths))

                # _record_histogram('ExternalDiscountedReturns', np.asarray(discounted_returns))
                # _record_histogram('ExternalUndiscountedReturns', np.asarray(undiscounted_returns))

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
                if self.n_epochs_per_eval != 0 and runner.step_itr % self.n_epochs_per_eval == 0:
                    self._evaluate_policy(runner)

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
                agent_update=self._get_policy_param_values(policy_sampler_key),
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
        if (runner.step_itr + 2) % self.n_epochs_per_log == 0 and wandb.run is not None:
            fig, ax = plt.subplots()
            env = runner._env
            env.draw(ax)
            list_viz_traj = []
            for i in range(len(trajectories)):
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
            wandb.log(({"train_Maze_traj": wandb.Image(filepath)}), step=runner.step_itr)
            
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

    def process_samples(self, paths):       # 【修改】我需要再这里做修改，修改buffer的训练元组；
        data = defaultdict(list)
        for i in range(len(paths)):                  # 【效率】一条一条处理，不知道能不能向量化？变成tensor按照矩阵处理？
            path = paths[i]
            data['obs'].append(path['observations'])
            data['next_obs'].append(path['next_observations'])
            data['actions'].append(path['actions'])
            data['rewards'].append(path['rewards'])
            data['dones'].append(path['dones'])
            data['returns'].append(tensor_utils.discount_cumsum(path['rewards'], self.discount))
            # 好像用不到这个ori_obs，用[1]列表代替；
            # data['ori_obs'].append(path['env_infos']['ori_obs'])
            # data['next_ori_obs'].append(path['env_infos']['next_ori_obs'])
            data['ori_obs'].append(path['observations'])
            data['next_ori_obs'].append(path['next_observations'])
            if 'pre_tanh_value' in path['agent_infos']:
                data['pre_tanh_values'].append(path['agent_infos']['pre_tanh_value'])
            if 'log_prob' in path['agent_infos']:
                data['log_probs'].append(path['agent_infos']['log_prob'])
            if 'option' in path['agent_infos']:
                data['options'].append(path['agent_infos']['option'])
                data['next_options'].append(np.concatenate([path['agent_infos']['option'][1:], path['agent_infos']['option'][-1:]], axis=0))
            '''
            zhanghe:
            add goal into the turple;
            1. random sample from the future traj.;
            2. using the final one;
            3. together;
            '''
            # Method 2:
            
            # data['goal'].append(np.tile(path['observations'][-1], (path['observations'].shape[0], 1)))       # 不知道最后一个需不需要特殊在意，感觉问题不大；
            # traj_len = len(path['observations'])
            # now_index = np.arange(traj_len)
            # random_index = np.random.randint(1, traj_len-1, size=traj_len)
            # future_index = np.minimum(now_index + random_index, traj_len-1)
            # data['sub_goal'].append(path['observations'][future_index])
            
            # 0717: 随机截断，作为subgoal
            # start = 0
            # traj_len = len(path['observations'])
            # min_sub_traj_len = int(traj_len / 5)
            # split_index_1 = np.random.randint(min_sub_traj_len, traj_len - 3*min_sub_traj_len)
            # split_index_2 = np.random.randint(split_index_1 + min_sub_traj_len, traj_len - min_sub_traj_len)
            
            # path_subgoal = np.zeros(path['observations'].shape)
            # path_subgoal[0:split_index_1+1] = np.tile(path['observations'][split_index_1], (split_index_1+1, 1)) 
            # path_subgoal[split_index_1+1:split_index_2+1] = np.tile(path['observations'][split_index_2], (split_index_2 - split_index_1, 1)) 
            # path_subgoal[split_index_2+1:] = np.tile(path['observations'][-1], (traj_len-1-split_index_2, 1)) 
            
            # data['sub_goal'].append(path_subgoal)
            
            if 'sub_goal' in path['agent_infos']:
                data['sub_goal'].append(path["agent_infos"]["sub_goal"])
                # 0718：随机选择subgoal,重采样整段作为subgoal；
                traj_len = len(path['observations'])
                if traj_len > 100:
                    subgoal_indices = np.random.choice(traj_len, 2, replace=False)
                    for j in range(len(subgoal_indices)):
                        subgoal_index = subgoal_indices[j]
                        data['obs'].append(path['observations'][:subgoal_index+1])
                        data['next_obs'].append(path['next_observations'][:subgoal_index+1])
                        data['actions'].append(path['actions'][:subgoal_index+1])
                        data['rewards'].append(path['rewards'][:subgoal_index+1])
                        path['dones'][subgoal_index] = 1
                        data['dones'].append(path['dones'][:subgoal_index+1])
                        data['returns'].append(tensor_utils.discount_cumsum(path['rewards'][:subgoal_index+1], self.discount))
                        # 好像用不到这个ori_obs，用[1]列表代替；
                        # data['ori_obs'].append(path['env_infos']['ori_obs'])
                        # data['next_ori_obs'].append(path['env_infos']['next_ori_obs'])
                        data['ori_obs'].append(path['observations'][:subgoal_index+1])
                        data['next_ori_obs'].append(path['next_observations'][:subgoal_index+1])
                        if 'pre_tanh_value' in path['agent_infos']:
                            data['pre_tanh_values'].append(path['agent_infos']['pre_tanh_value'][:subgoal_index+1])
                        if 'log_prob' in path['agent_infos']:
                            data['log_probs'].append(path['agent_infos']['log_prob'][:subgoal_index+1])
                        if 'option' in path['agent_infos']:
                            data['options'].append(path['agent_infos']['option'][:subgoal_index+1])
                            data['next_options'].append(np.concatenate([path['agent_infos']['option'][:subgoal_index+1][1:], path['agent_infos']['option'][:subgoal_index+1][-1:]], axis=0))
                        data['sub_goal'].append(np.tile(path['observations'][:subgoal_index+1][-1], (subgoal_index+1, 1)))
                    
                
            
        return data


    def _get_policy_param_values(self, key):
        param_dict = self.policy[key].get_param_values()
        for k in param_dict.keys():
            if self.sample_cpu:
                param_dict[k] = param_dict[k].detach().cpu()
            else:
                param_dict[k] = param_dict[k].detach()
        return param_dict

    def _generate_option_extras(self, options, sub_goal=None):
        extras = [{"option": option, "exploration_type": 0} for option in options]
        if sub_goal is not None:
            for i in range(len(sub_goal)):
                extras[i]["sub_goal"] = sub_goal[i]
                
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
