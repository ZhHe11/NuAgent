import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians

import wandb
from iod.agent import AgentWrapper


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
        
        # added 
        self.last_w = None
        self.epoch_data = None
        
        '''
        wrapper for agent for online interaction.
        '''
        policy_for_agent = {
            "default_policy": self.option_policy,
            # "traj_encoder": self.traj_encoder,
            # "target_traj_encoder": self.target_traj_encoder,
        }
        self.policy_for_agent = AgentWrapper(policies=policy_for_agent) 
        
        

    @property
    def policy(self):
        return {
            'option_policy': self.policy_for_agent,
        }

    @torch.no_grad()
    def get_TS(self, s, s_next, Support, is_norm=True):
        '''
        s = [batch, dim]
        a = [batch, dim]
        batch_regret = [batch]
        '''
        # s = s.reshape(s.shape[0]*s.shape[1], s.shape[-1])   # [batch*seq, dim]
        # s_next = s_next.reshape(s_next.shape[0]*s_next.shape[1], s_next.shape[-1])  # [batch*seq, dim]
        phi_s = self.traj_encoder(s).mean
        phi_s_next = self.traj_encoder(s_next).mean
        TS = torch.zeros(self.dim_option).to(self.device)
        for i in range(self.dim_option):
            TS[i] = ((phi_s_next - phi_s) * Support[i].unsqueeze(0)).sum(dim=-1).mean()    # [batch*seq, 1]
        TS_mean = TS.mean()
        if is_norm:
            TS = (TS - TS.mean()) / (TS.std() + 1e-8)

        return TS, TS_mean


    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self, runner):
        if self.discrete:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
            
            # explore type
            explore_type = 'baseline'
            if explore_type == 'table' and self.epoch_data is not None:
                import random
                import torch.nn.functional as F

                std = 0.1
                alpha = 0.1
                if self.last_w is None:
                    self.Support_tensor = torch.eye(self.dim_option).to(self.device)
                    Weight_important = torch.ones(self.dim_option).to(self.device) / self.dim_option
                    self.last_TS = torch.zeros(self.dim_option).to(self.device)
                
                v = self.epoch_data
                TS, TS_mean = self.get_TS(s=v['obs'], s_next=v['next_obs'], Support=self.Support_tensor, is_norm=False)
                Weight_important = torch.clip(F.softmax(TS - self.last_TS, dim=-1), min= 1/(2*self.dim_option))
                Weight_important = Weight_important / Weight_important.sum()
                
                w = Weight_important.cpu().numpy()
                Sample_z = random.choices(self.Support_tensor.cpu().numpy(), weights=w, k=self.num_random_trajectories)
                Sample_z_array = np.array(Sample_z)
                Sample_z_array = Sample_z_array + std * np.random.randn(Sample_z_array.shape[0], Sample_z_array.shape[1])

                Sample_z_array = Sample_z_array / np.linalg.norm(Sample_z_array, axis=-1, keepdims=True)
                extras = self._generate_option_extras(Sample_z_array)
                print('Weight_important:', w)
            
                self.last_TS = self.last_TS + alpha * (TS - self.last_TS)
            
        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            extras = self._generate_option_extras(random_options)

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

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data

    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)

        epoch_data = self._flatten_data(path_data)

        tensors = self._train_components(epoch_data)

        return tensors

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()
                # self.epoch_data = v

            self._optimize_te(tensors, v)
            self._update_rewards(tensors, v)
            self._optimize_op(tensors, v)

        return tensors

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

    def _update_rewards(self, tensors, v):
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

    def _update_loss_te(self, tensors, v):
        self._update_rewards(tensors, v)
        rewards = v['rewards']

        obs = v['obs']
        next_obs = v['next_obs']

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
            phi_x = v['cur_z']
            phi_y = v['next_z']

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

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
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
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
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
            self.eval_kitchen_metra(runner)
            
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
        file_name = path + 'traj_encoder.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'traj_encoder': self.traj_encoder,
        }, file_name)

    def eval_kitchen_metra(self, runner):
        random_options = np.eye(self.dim_option)
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
        