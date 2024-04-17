# refer to: https://github.com/seohongpark/METRA/blob/master/iod/metra.py
from typing import Dict
from argparse import Namespace

import copy
import math
import torch
import numpy as np
from torch import nn

from application.hilp.learner import HILPAgent
from . import sac_utils
from .networks import ContinuousMLPQFunctionEx, ParameterModule, GaussianMLP, Policy

# configuration needed:
#   use_inner_product: bool
#   discrete_goal: bool
#   dual_dist: str, 's2_from_s',
#   dual_reg: bool, enable regularizer or not
#   alpha: float
class MetraAgent(HILPAgent):
    def __init__(
        self,
        config: Namespace,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        load_path: str = None,
    ):
        super().__init__(config, obs_dim, goal_dim, act_dim, load_path)

    def create_networks(self, load_path: str = None) -> nn.ModuleDict:
        qf = ContinuousMLPQFunctionEx(self.obs_dim, self.act_dim, hidden_dims=self.config.value_hidden_dims, ensemble_num=2)
        option_actor = Policy(self.obs_dim, self.goal_dim, self.act_dim, hidden_dims=self.config.value_hidden_dims)
        log_alpha = ParameterModule(torch.Tensor([np.log(self.config.alpha)]))
        dual_lam = ParameterModule(torch.Tensor([np.log(self.config.dual_lam)]))
        traj_encoder = GaussianMLP(self.obs_dim, 0, self.config.option_dim)

        networks = {
            'qf': qf,
            'target_qf': copy.deepcopy(qf),
            'option_actor': option_actor,
            'traj_encoder': traj_encoder,
            'dual_lam': dual_lam,
            'log_alpha': log_alpha
        }

        if self.config.algo == 'dads':
            skill_dynamics = GaussianMLP(self.obs_dim + self.config.option_dim, 0, self.obs_dim, log_std_min=math.log(0.3), log_std_max=math.log(10))
            networks['skill_dynamics'] = skill_dynamics

        if self.config.use_dist_predictor:
            networks['dist_predictor'] = GaussianMLP(self.obs_dim, 0, self.obs_dim, log_std_min=-6, log_std_max=6)
        return nn.ModuleDict(networks)
    
    def setup_optimizer(self):
        optimizer = {
            'qf': torch.optim.Adam(self.networks['qf'].parameters(), lr=self.config.lr),
            'option_policy': torch.optim.Adam(self.option_policy.parameters(), lr=self.config.lr_op),
            'traj_encoder': torch.optim.Adam(self.traj_encoder.parameters(), lr=self.config.lr_te),
            'dual_lam': torch.optim.Adam(self.dual_lam.parameters(), lr=self.config.lr_dual)
        }

        if self.config.algo == 'dads':
            optimizer['skill_dynamics'] = torch.optim.Adam(self.networks['skill_dynamics'].parameters(), lr=self.config.lr_te)

        if self.config.use_dist_predictor:
            optimizer['dist_predictor'] = torch.optim.Adam(self.networks['dist_predictor'].parameters(), lr=self.config.lr_op)

    @property
    def log_alpha(self) -> nn.Module:
        return self.networks['log_alpha']

    @property
    def dual_lam(self) -> nn.Module:
        return self.networks['dual_lam']

    @property
    def qf(self) -> nn.Module:
        return self.networks['qf']
    
    @property
    def target_qf(self) -> nn.Module:
        return self.networks['target_qf']
    
    @property
    def traj_encoder(self) -> nn.Module:
        return self.networks['traj_encoder']
    
    @property
    def dist_predictor(self) -> nn.Module:
        return self.networks['dist_predictor']

    @property
    def option_policy(self) -> nn.Module:
        return self.networks['option_actor']

    def cal_rewards(
        self, loss_dict: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ):
        # batch requires:
        #   1. options: goals
        obs = batch["observations"]
        next_obs = batch["next_observations"]

        if self.config.use_inner_product:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            target_z = next_z - cur_z

            # for discrete environment only now
            if self.config.discrete_goal:
                masks = (
                    (batch["options"] - batch["options"].mean(dim=1, keepdim=True))
                    * self.dim_option
                    / (self.dim_option - 1 if self.dim_option != 1 else 1)
                )
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * batch["options"]).sum(dim=1)
                rewards = inner

            batch.update({"cur_z": cur_z, "next_z": next_z})
        else:
            target_dists = self.traj_encoder(next_obs)
            if self.config.discrete_goal:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(
                    logits, batch["options"].argmax(dim=1), reduce="none"
                )
            else:
                rewards = target_dists.log_prob(batch["options"])

        batch["rewards"] = rewards
        loss_dict.update(
            {"PureRewardMean": rewards.mean(), "UreRewardStd": rewards.std()}
        )

        return rewards

    def compute_loss_dual_lam(
        self, loss_dict: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (batch["cst_penalty"].detach()).mean()

        loss_dict.update(
            {
                "DualLam": dual_lam,
                "LossDualLam": loss_dual_lam,
            }
        )

    def compute_loss_qf(
        self, loss_dict: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ):
        processed_cat_obs = self._get_concat_obs(
            self.option_policy.process_observations(batch["observations"]),
            batch["options"],
        )
        next_processed_cat_obs = self._get_concat_obs(
            self.option_policy.process_observations(batch["next_obs"]),
            batch["next_options"],
        )

        sac_utils.compute_loss_qf(
            self,
            loss_dict,
            batch,
            obs=processed_cat_obs,
            actions=batch["actions"],
            next_obs=next_processed_cat_obs,
            dones=batch["dones"],
            rewards=batch["rewards"] * self._reward_scale_factor,
            policy=self.option_policy,
        )

        batch.update(
            {
                "processed_cat_obs": processed_cat_obs,
                "next_processed_cat_obs": next_processed_cat_obs,
            }
        )

    def _get_concat_obs(self, obs):
        raise NotImplementedError
    
    def _target_entropy

    def compute_loss_op(
        self, loss_dict: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ):
        processed_cat_obs = self._get_concat_obs(
            self.option_policy.process_observations(batch["observations"]),
            batch["options"],
        )
        sac_utils.compute_loss_sacp(
            self,
            loss_dict,
            batch,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def compute_loss_alpha(
        self, loss_dict: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ):
        sac_utils.compute_loss_alpha(
            self,
            loss_dict,
            batch,
        )

    def compute_loss(
        self, loss_dict: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ):
        rewards = self.cal_rewards(loss_dict, batch)

        obs = batch["observations"]
        next_obs = batch["next_observations"]

        if self.config.dual_dist == "s2_from_s":
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            loss_dict.update({"LossDp": loss_dp})

        if self.config.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            # retrieve the representation corresponding to the two states
            phi_x = batch["cur_z"]
            phi_y = batch["next_z"]

            if self.config.dual_dist == "l2":
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.config.dual_dist == "one":
                cst_dist = torch.ones_like(x[:, 0])
            elif self.config.dual_dist == "s2_from_s":
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1.0 / s2_dist_std
                geo_mean = torch.exp(
                    torch.log(scaling_factor).mean(dim=1, keepdim=True)
                )
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(
                    torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor,
                    dim=1,
                )

                loss_dict.update(
                    {
                        "ScalingFactor": scaling_factor.mean(dim=0),
                        "NormalizedScalingFactor": normalized_scaling_factor.mean(
                            dim=0
                        ),
                    }
                )
            else:
                raise NotImplementedError

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            te_obj = rewards + dual_lam.detach() * cst_penalty

            batch.update({"cst_penalty": cst_penalty})
            loss_dict.update(
                {
                    "DualCstPenalty": cst_penalty.mean(),
                }
            )
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        loss_dict.update(
            {
                "TeObjMean": te_obj.mean(),
                "LossTe": loss_te,
            }
        )
