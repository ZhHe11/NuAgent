#!/usr/bin/env python3
import tempfile

import dowel_wrapper

assert dowel_wrapper is not None
import dowel

import wandb

import argparse
import datetime
import functools
import os
import sys
import platform
import torch.multiprocessing as mp

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import better_exceptions
import numpy as np

better_exceptions.hook()

import torch

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.distributions import TanhNormal

from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.experiment.option_local_runner import OptionLocalRunner
from garagei.envs.consistent_normalized_env import consistent_normalize
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.modules.with_encoder import WithEncoder, Encoder
from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx, GaussianMLPIndependentStdModuleEx, GaussianMLPModuleEx
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import xavier_normal_ex
from iod.metra import METRA
from iod.dads import DADS
from iod.utils import get_normalizer_preset

from tests.make_env import make_env

'''
我想写一个脚本，用来可视化某一个环境某一个任务下的agent的行为
1. 加载我训练好的模型；
2. 画表征z的轨迹图；
3. 
'''

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--run_group', type=str, default='Debug')
    # parser.add_argument('--normalizer_type', type=str, default='off', choices=['off', 'preset'])
    parser.add_argument('--encoder', type=int, default=0)

    parser.add_argument('--env', type=str, default='maze', choices=[
        'maze', 'half_cheetah', 'ant', 'dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'kitchen',
    ])
    parser.add_argument('--frame_stack', type=int, default=None)

    parser.add_argument('--max_path_length', type=int, default=50)

    # parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1])
    # parser.add_argument('--sample_cpu', type=int, default=1, choices=[0, 1])
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--n_parallel', type=int, default=4)
    # parser.add_argument('--n_thread', type=int, default=1)

    # parser.add_argument('--n_epochs', type=int, default=1000000)
    # parser.add_argument('--traj_batch_size', type=int, default=8)
    # parser.add_argument('--trans_minibatch_size', type=int, default=256)
    # parser.add_argument('--trans_optimization_epochs', type=int, default=200)

    # parser.add_argument('--n_epochs_per_eval', type=int, default=125)
    # parser.add_argument('--n_epochs_per_log', type=int, default=25)
    # parser.add_argument('--n_epochs_per_save', type=int, default=1000)
    # parser.add_argument('--n_epochs_per_pt_save', type=int, default=1000)
    # parser.add_argument('--n_epochs_per_pkl_update', type=int, default=None)
    # parser.add_argument('--num_random_trajectories', type=int, default=48)
    # parser.add_argument('--num_video_repeats', type=int, default=2)
    # parser.add_argument('--eval_record_video', type=int, default=1)
    # parser.add_argument('--eval_plot_axis', type=float, default=None, nargs='*')
    # parser.add_argument('--video_skip_frames', type=int, default=1)

    # parser.add_argument('--dim_option', type=int, default=2)

    # parser.add_argument('--common_lr', type=float, default=1e-4)
    # parser.add_argument('--lr_op', type=float, default=None)
    # parser.add_argument('--lr_te', type=float, default=None)

    # parser.add_argument('--alpha', type=float, default=0.01)

    # parser.add_argument('--algo', type=str, default='metra', choices=['metra', 'dads'])

    # parser.add_argument('--sac_tau', type=float, default=5e-3)
    # parser.add_argument('--sac_lr_q', type=float, default=None)
    # parser.add_argument('--sac_lr_a', type=float, default=None)
    # parser.add_argument('--sac_discount', type=float, default=0.99)
    # parser.add_argument('--sac_scale_reward', type=float, default=1.)
    # parser.add_argument('--sac_target_coef', type=float, default=1.)
    # parser.add_argument('--sac_min_buffer_size', type=int, default=10000)
    # parser.add_argument('--sac_max_buffer_size', type=int, default=300000)

    # parser.add_argument('--spectral_normalization', type=int, default=0, choices=[0, 1])

    # parser.add_argument('--model_master_dim', type=int, default=1024)
    # parser.add_argument('--model_master_num_layers', type=int, default=2)
    # parser.add_argument('--model_master_nonlinearity', type=str, default=None, choices=['relu', 'tanh'])
    # parser.add_argument('--sd_const_std', type=int, default=1)
    # parser.add_argument('--sd_batch_norm', type=int, default=1, choices=[0, 1])

    # parser.add_argument('--num_alt_samples', type=int, default=100)
    # parser.add_argument('--split_group', type=int, default=65536)

    # parser.add_argument('--discrete', type=int, default=0, choices=[0, 1, 2])
    # parser.add_argument('--inner', type=int, default=1, choices=[0, 1])
    # parser.add_argument('--unit_length', type=int, default=1, choices=[0, 1])  # Only for continuous skills

    # parser.add_argument('--dual_reg', type=int, default=1, choices=[0, 1])
    # parser.add_argument('--dual_lam', type=float, default=30)
    # parser.add_argument('--dual_slack', type=float, default=1e-3)
    # parser.add_argument('--dual_dist', type=str, default='one', choices=['l2', 's2_from_s', 'one'])
    # parser.add_argument('--dual_lr', type=float, default=None)

    return parser


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # define the params
    args = get_argparser().parse_args()
    args.env = 'kitchen'
    args.max_path_length = 50
    args.frame_stack = 4
    args.encoder = 1
    args.normalizer_type = 'off'
    args.seed = 0

    env = make_env(args, args.max_path_length)

    # show the env
    env.reset()
    for i in range(100):

        print(env.step(env.action_space.sample()))

        env.get_state()

        env.render()









