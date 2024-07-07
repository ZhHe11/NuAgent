import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians


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

        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
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

        tensors = self._train_components(epoch_data)    # 训练模型，知识我不理解，为什么还要返回tensor，tensor后续没再使用;

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

    def _optimize_te(self, tensors, internal_vars):         # 【loss】对与skill表征的loss:
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

    def _update_rewards(self, tensors, v):                      # 【修改】这里修改reward的计算方法；
        obs = v['obs']
        next_obs = v['next_obs']

        if self.inner:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            target_z = next_z - cur_z

            if self.discrete:
                masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * v['options']).sum(dim=1)
                rewards = inner

            # For dual objectives
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })
        else:
            target_dists = self.traj_encoder(next_obs)

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['options'])

        tensors.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        v['rewards'] = rewards

    def _update_loss_te(self, tensors, v):          # 【更新】要修改表征loss的计算方法；
        self._update_rewards(tensors, v)            # 为什么要更新reward？reward其实就是计算技能表征z与当前运动方向的内积；
        rewards = v['rewards']

        '''
        zhanghe 
        我要加入goal，重新计算两个表征z_sample和z_start_goal之间的距离；
        '''
        z_sample = v['options']
        phi_g = self.traj_encoder(v['goal']).mean
        phi_s = self.traj_encoder(v['obs']).mean
        phi_s_next = self.traj_encoder(v['next_obs']).mean

        # 暂时使用表征做差：
        z_start_next = phi_s_next - phi_s
        z_start_goal = phi_g - phi_s
        z_next_goal = phi_g - phi_s_next
        skill_discount = 0.5
        # new_reward = (z_start_next * z_sample).sum(dim=1) + skill_discount * (z_start_goal * z_sample).sum(dim=1) 

        ## 【ctb】len_weight:
        # 我想让离final越近的权重越大，离final越远的权重越小；
        # 方法1： 用step作为约束；
        # len_weight = 

        # 方法2： 用与s_final的相似度作为约束；
        min_val = 1e-2
        max_val = 1e2
        len_weight = 1 / (torch.norm(z_start_goal, p=2).detach() + 1e-3)
        len_weight = torch.clamp(len_weight, min=1e-2, max=1e2)
        norm_len_weight = (len_weight - min_val) / (max_val - min_val)

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
            
            # 【ori】原方法：
            te_obj = rewards + dual_lam.detach() * cst_penalty                      # 这是最终的loss： reward + 惩罚项；
            
            # 【ctb】增加s_final和z_sample的约束，to be finished
            # te_obj = new_reward + dual_lam.detach() * cst_penalty    

            # 【ctb】增加len_weight
            # te_obj = norm_len_weight * rewards + dual_lam.detach() * cst_penalty    

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

    def _update_loss_qf(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['next_options'])

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=v['rewards'] * self._reward_scale_factor,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
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
        if self.discrete == 1: # self.discrete == 1 是原来的方法，当维度较大的时候，效果比较好，能强行分出不同的技能；
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):       # 重复多次，
                    random_options.append(eye_options[i])   # 把每次的option都存下来，在dis=1的时候，所有的option都相同；
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
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)     # [48，2]全部随机
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

        # Plotting: 画一整条轨迹在表征空间的分布；
        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        option_dists = self.traj_encoder(last_obs)          # 对每条轨迹的最后一个状态进行encode编码；

        option_means = option_dists.mean.detach().cpu().numpy()   
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

        option_colors = random_option_colors

        # 画option在表征空间的分布；
        # option_dists是最后一个状态（goal）的表征分布；
        # 所以，这个图越发散，说明最终状态的表征分布越分散，说明不同的z_sample能指向不同的goal；
        # 但是训练时引入了goal，反而发散程度减弱，发散速度减弱，为什么？按理会朝着goal的方向收敛，更直，有没有可能是ant环境比较简单，已经很直了；
        # 所谓直，就是不饶弯路，直接到达目标，让表征学习的更加准确；
        # 再修改一下policy看看， 毕竟her主要是让policy有更多的训练样例，在错误中学习正确的东西（her的insight）；
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
                
            # video的数据是另外再收集的，重新定义了option且是随机的；option是2维的话会限定各个角度都有；
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
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)
