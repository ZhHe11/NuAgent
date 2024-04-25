# refer to: https://github.com/seohongpark/METRA/blob/master/iod/metra.py
from typing import Dict
from argparse import Namespace

import copy
import math

import gym
import torch
import numpy as np

from torch import nn
from gym.spaces import Discrete

from uniagent.data.replay_buffer import ReplayBuffer
from application.hilp.learner import HILPAgent
from . import sac_utils
from .networks import ContinuousMLPQFunctionEx, ParameterModule, GaussianMLP, Policy
from .exp_utils import ExpManager


def create_replay_buffer(args: Namespace, env: gym.Env) -> ReplayBuffer:
    return ReplayBuffer(
        args.buffer_size,
        shape_and_dtypes={
            "observation": (env.observation_space.shape, np.float32),
            "action": ((env.action_space.n,), int)
            if isinstance(env.action_space, Discrete)
            else (env.action_space.shape, np.float32),
            "reward": ((), np.float32),
            "done": ((), bool),
            # always in vector
            "option": ((args.option_dim,), np.float32),
            "next_observation": (env.observation_space.shape, np.float32),
            "next_option": ((args.option_dim,), np.float32),
        },
    )


class RandomOptionPlanner(nn.Module):
    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)

    @torch.no_grad()
    def sample_option(self, observation: np.ndarray):
        return self(observation).cpu().numpy()

    def forward(self, observation: torch.Tensor = None) -> torch.Tensor:
        if self.config.discrete_option:
            z = np.zeros(self.config.option_dim, dtype=np.float32)
            idx = np.random.choice(self.config.option_dim)
            z[idx] = 1.0
        else:
            z = np.random.randn(self.config.option_dim)
        return torch.from_numpy(z).to(self.config.device)


# configuration needed:
#   use_inner_product: bool
#   discrete_option: bool
#   dual_dist: str, 's2_from_s',
#   dual_reg: bool, enable regularizer or not
#   alpha: float
class MetraAgent(HILPAgent, ExpManager):
    def __init__(
        self,
        config: Namespace,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        action_space: gym.Space,
        load_path: str = None,
    ):
        HILPAgent.__init__(self, config, obs_dim, goal_dim, act_dim, load_path)
        ExpManager.__init__(self)
        self.action_space = action_space
        self.target_entropy = (
            -np.prod(self.action_space.shape).item() / 2.0 * self.config.sac_target_coef
        )

    def create_networks(self, load_path: str = None) -> nn.ModuleDict:
        qf = ContinuousMLPQFunctionEx(
            self.obs_dim + self.goal_dim,
            self.act_dim,
            hidden_dims=self.config.value_hidden_dims,
            ensemble_num=2,
        )
        option_conditioned_policy = Policy(
            self.obs_dim,
            self.goal_dim,
            self.act_dim,
            hidden_dims=self.config.actor_hidden_dims,
        )
        option_planner = self.create_option_planner()
        log_alpha = ParameterModule(torch.Tensor([np.log(self.config.alpha)]))
        dual_lam = ParameterModule(torch.Tensor([np.log(self.config.dual_lam)]))
        # for observation representation learning
        traj_encoder = GaussianMLP(self.obs_dim, 0, self.config.option_dim)
        dist_predictor = self.create_dist_predictor()
        skill_dynamics = self.create_skill_dynamics_model()

        networks = {
            "qf": qf,
            "target_qf": copy.deepcopy(qf),
            "option_policy": option_conditioned_policy,
            # the phi, for state embedding
            "traj_encoder": traj_encoder,
            "dual_lam": dual_lam,
            "log_alpha": log_alpha,
            "option_planner": option_planner,
            "dist_predictor": dist_predictor,
            "skill_dynamics": skill_dynamics,
        }

        return nn.ModuleDict(networks)

    def forward(
        self,
        method: str,
        observations: torch.Tensor,
        options: torch.Tensor = None,
        actions: torch.Tensor = None,
    ) -> torch.Tensor:
        if method in ["option_policy"]:
            return self.networks[method](torch.concat([observations, options], dim=-1))
        else:
            return self.networks[method](observations, actions)

    def create_dist_predictor(self):
        if self.config.use_dist_predictor:
            return GaussianMLP(
                self.obs_dim, 0, self.obs_dim, log_std_min=-6, log_std_max=6
            )
        else:
            return None

    def create_skill_dynamics_model(self):
        if self.config.algo == "dads":
            skill_dynamics = GaussianMLP(
                self.obs_dim + self.config.option_dim,
                0,
                self.obs_dim,
                log_std_min=math.log(0.3),
                log_std_max=math.log(10),
            )
            return skill_dynamics
        else:
            return None

    def create_option_planner(self):
        """Create option planner for option generation, defaults to random model, you can override it"""

        if self.config.use_option_planner:
            return RandomOptionPlanner()
        else:
            return None

    def setup_optimizer(self):
        optimizer = {
            "qf": torch.optim.Adam(
                self.networks["qf"].parameters(), lr=self.config.sac_lr_q
            ),
            "option_policy": torch.optim.Adam(
                self.option_policy.parameters(), lr=self.config.sac_lr_a
            ),
            "traj_encoder": torch.optim.Adam(
                self.traj_encoder.parameters(), lr=self.config.lr_te
            ),
            "dual_lam": torch.optim.Adam(
                self.dual_lam.parameters(), lr=self.config.lr_dual
            ),
            "log_alpha": torch.optim.Adam(
                self.log_alpha.parameters(), lr=self.config.sac_lr_a
            ),
        }

        if self.config.algo == "dads":
            optimizer["skill_dynamics"] = torch.optim.Adam(
                self.networks["skill_dynamics"].parameters(), lr=self.config.lr_te
            )

        if self.config.use_dist_predictor:
            optimizer["dist_predictor"] = torch.optim.Adam(
                self.networks["dist_predictor"].parameters(), lr=self.config.lr_op
            )

        if self.config.use_option_planner:
            optimizer["option_planner"] = torch.optim.Adam(
                self.networks["option_planner"].parameters(), lr=self.config.common_lr
            )

        return optimizer

    @property
    def log_alpha(self) -> nn.Module:
        return self.networks["log_alpha"]

    @property
    def dual_lam(self) -> nn.Module:
        return self.networks["dual_lam"]

    @property
    def qf(self) -> nn.Module:
        return self.networks["qf"]

    @property
    def target_qf(self) -> nn.Module:
        return self.networks["target_qf"]

    @property
    def traj_encoder(self) -> nn.Module:
        return self.networks["traj_encoder"]

    @property
    def dist_predictor(self) -> nn.Module:
        return self.networks["dist_predictor"]

    @property
    def option_policy(self) -> nn.Module:
        return self.networks["option_policy"]

    @property
    def option_planner(self) -> nn.Module:
        return self.networks["option_planner"]

    @torch.no_grad()
    def sample_option(self, observation: np.ndarray) -> np.ndarray:
        if self.config.use_option_planner:
            observation = torch.from_numpy(observation).float().to(self.config.device)
            option = self.option_planner.sample_option(observation)
        else:
            option = np.random.randn(self.config.option_dim)
        return option

    @torch.no_grad()
    def sample_action(self, observation: np.ndarray, option: np.ndarray) -> np.ndarray:
        observation = torch.from_numpy(observation).float().to(self.config.device)
        option = torch.from_numpy(option).float().to(self.config.device)
        dist: torch.distributions.Distribution = self(
            "option_policy", observation, option
        )
        action = dist.sample()
        return action.cpu().numpy()

    def cal_rewards(self, batch: Dict[str, torch.Tensor]):
        obs = batch["observation"]
        next_obs = batch["next_observation"]

        if self.config.use_inner_product:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            target_z = next_z - cur_z

            # for discrete environment only now
            if self.config.discrete_option:
                masks = (
                    (batch["option"] - batch["option"].mean(dim=1, keepdim=True))
                    * self.dim_option
                    / (self.dim_option - 1 if self.dim_option != 1 else 1)
                )
                rewards = (target_z * masks).sum(dim=1)
            else:
                assert target_z.shape == batch["option"].shape, (
                    target_z.shape,
                    batch["option"].shape,
                )
                rewards = (target_z * batch["option"]).sum(dim=1)

            batch.update({"cur_z": cur_z, "next_z": next_z})
        else:
            target_dists = self.traj_encoder(next_obs)
            if self.config.discrete_option:
                logits = target_dists.mean
                import pdb

                pdb.set_trace()
                rewards = -torch.nn.functional.cross_entropy(
                    logits, batch["option"].argmax(dim=1), reduce="none"
                )
            else:
                rewards = target_dists.log_prob(batch["option"])

        info = {
            "PureRewardMean": rewards.mean().cpu().item(),
            "UreRewardStd": rewards.std().cpu().item(),
        }

        return rewards, info

    def compute_loss_dual_lam(self, batch: Dict[str, torch.Tensor]):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (batch["cst_penalty"].detach()).mean()

        return loss_dual_lam, {
            "DualLam": dual_lam.detach().mean().cpu().item(),
            "LossDualLam": loss_dual_lam.detach().cpu().item(),
        }

    def compute_loss_qf(self, action_space: gym.Space, batch: Dict[str, torch.Tensor]):
        processed_cat_obs = self._get_concat_obs(
            self.option_policy.process_observations(batch["observation"]),
            batch["option"],
        )
        # FIXME(ming): use next_option, not option
        next_processed_cat_obs = self._get_concat_obs(
            self.option_policy.process_observations(batch["next_observation"]),
            batch["next_option"],
        )

        loss, info = sac_utils.compute_loss_qf(
            self,
            action_space=action_space,
            batch=batch,
            obs=processed_cat_obs,
            actions=batch["action"],
            next_obs=next_processed_cat_obs,
            dones=batch["done"],
            rewards=batch["reward"] * self.config.sac_scale_reward,
            policy=self.option_policy,
        )

        batch.update(
            {
                "processed_cat_obs": processed_cat_obs,
                "next_processed_cat_obs": next_processed_cat_obs,
            }
        )

        return loss, info

    def _get_concat_obs(self, obs, option):
        assert (
            len(obs.shape) == len(option.shape) and obs.shape[0] == option.shape[0]
        ), (obs.shape, option.shape)
        return torch.concat([obs, option], dim=-1)

    def compute_loss_op(self, batch: Dict[str, torch.Tensor]):
        processed_cat_obs = self._get_concat_obs(
            self.option_policy.process_observations(batch["observation"]),
            batch["option"],
        )
        loss, info = sac_utils.compute_loss_sacp(
            self,
            batch,
            obs=processed_cat_obs,
            policy=self.option_policy,
            action_space=self.action_space,
        )
        return loss, info

    def compute_loss_alpha(self, batch: Dict[str, torch.Tensor]):
        loss, info = sac_utils.compute_loss_alpha(
            self,
            batch,
        )

        return loss, info

    def compute_loss_te(self, batch: Dict[str, torch.Tensor]):
        loss_info = {}

        # compute reward for each time step in the batch
        rewards, info = self.cal_rewards(batch)
        loss_info.update(info)

        obs = batch["observation"]
        next_obs = batch["next_observation"]

        if self.config.dual_dist == "s2_from_s":
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            loss_info["LossDp"] = loss_dp.cpu().item()
        else:
            loss_dp = 0

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

                loss_info.update(
                    {
                        "ScalingFactor": scaling_factor.mean(dim=0).item(),
                        "NormalizedScalingFactor": normalized_scaling_factor.mean(
                            dim=0
                        ).item(),
                    }
                )
            else:
                raise NotImplementedError

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.config.dual_slack)
            te_obj = rewards + dual_lam.detach() * cst_penalty

            batch["cst_penalty"] = cst_penalty
            loss_info["DualCstPenalty"] = cst_penalty.mean().cpu().item()
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()
        loss_info["TeObjMean"] = te_obj.mean().cpu().item()
        loss_info["LossTe"] = loss_te.cpu().item()

        return loss_dp + loss_te, loss_info

    def target_update(self):
        tau = self.config.tau
        for tp, p in zip(
            self.networks["target_qf"].parameters(), self.networks["qf"].parameters()
        ):
            tp.data.copy_(tau * p + (1 - tau) * tp)

    def step_optimizer(self, key: str = None):
        if key is not None:
            self.optimizer[key].step()
        else:
            for v in self.optimizer.values():
                v.step()

    def update(self, batch: Dict[str, np.ndarray], **kwargs) -> Dict[str, float]:
        batch = self.to_torch(batch)
        loss_info = {}

        self.zero_grad()
        te_loss, te_loss_info = self.compute_loss_te(batch)
        loss_info.update({f"TE/{k}": v for k, v in te_loss_info.items()})
        te_loss.backward()
        self.step_optimizer("traj_encoder")
        if self.config.use_dist_predictor:
            self.step_optimizer("dist_predictor")

        if self.config.dual_reg:
            self.zero_grad()
            lam_loss, lam_loss_info = self.compute_loss_dual_lam(batch)
            loss_info.update({f"LAM/{k}": v for k, v in lam_loss_info.items()})
            lam_loss.backward()
            self.step_optimizer("dual_lam")
        else:
            lam_loss = 0

        with torch.no_grad():
            batch["reward"], _ = self.cal_rewards(batch)

        self.zero_grad()
        qf_loss, qf_loss_info = self.compute_loss_qf(kwargs["action_space"], batch)
        qf_loss.backward()
        loss_info.update({f"QF/{k}": v for k, v in qf_loss_info.items()})
        self.step_optimizer("qf")

        # loss for option policy
        self.zero_grad()
        op_loss, op_loss_info = self.compute_loss_op(batch)
        op_loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm(
            self.option_policy.parameters(), 0.5
        )
        loss_info.update({f"OP/{k}": v for k, v in op_loss_info.items()})
        loss_info["OP/grad_norm"] = grad_norm.cpu().item()
        self.step_optimizer("option_policy")

        self.zero_grad()
        alpha_loss, alpha_loss_info = self.compute_loss_alpha(batch)
        alpha_loss.backward()
        loss_info.update({f"ALPHA/{k}": v for k, v in alpha_loss_info.items()})
        self.step_optimizer("log_alpha")

        self.zero_grad()
        self.target_update()

        return loss_info
