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
import imageio
from iod.metra import METRA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from tests.GetArgparser import get_argparser


'''
我想写一个脚本，用来可视化某一个环境某一个任务下的agent的行为
1. 加载我训练好的模型；
2. 画表征z的轨迹图；
3. 
'''

# save the traj. as fig
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
    path_file_traj = path + "traj" + ".png"
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('traj. in representation space')
    plt.legend()
    plt.savefig(path_file_traj)


def get_gaussian_module_construction(args,
                                     *,
                                     hidden_sizes,
                                     const_std=False,
                                     hidden_nonlinearity=torch.relu,
                                     w_init=torch.nn.init.xavier_uniform_,
                                     init_std=1.0,
                                     min_std=1e-6,
                                     max_std=None,
                                     **kwargs):
    module_kwargs = dict()
    if const_std:
        module_cls = GaussianMLPModuleEx
        module_kwargs.update(dict(
            learn_std=False,
            init_std=init_std,
        ))
    else:
        module_cls = GaussianMLPIndependentStdModuleEx
        module_kwargs.update(dict(
            std_hidden_sizes=hidden_sizes,
            std_hidden_nonlinearity=hidden_nonlinearity,
            std_hidden_w_init=w_init,
            std_output_w_init=w_init,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
        ))

    module_kwargs.update(dict(
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=hidden_nonlinearity,
        hidden_w_init=w_init,
        output_w_init=w_init,
        std_parameterization='exp',
        bias=True,
        spectral_normalization=args.spectral_normalization,
        **kwargs,
    ))
    return module_cls, module_kwargs


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # define the params
    # args = get_argparser().parse_args()
    # args.env = 'kichen'
    # args.max_path_length = 50
    # args.frame_stack = 3
    # args.encoder = 1
    # args.normalizer_type = 'off'
    # args.seed = 0
    # args.device = 'cuda:0'
    # device = args.device

    # maze
    args = get_argparser().parse_args()
    args.env = 'maze'
    args.max_path_length = 100
    # args.frame_stack = 3
    # args.encoder = 1
    args.normalizer_type = 'off'
    args.seed = 0
    args.device = 'cuda:7'
    args.render = 1 
    device = args.device



    # make env
    env = make_env(args, args.max_path_length)

    # # load model
    # load_option_policy = torch.load("/mnt/nfs2/zhanghe/project001/METRA/exp/Debug_Kitchen_original/sd000_1719919248_kitchen_metra/option_policy6000.pt")
    # load_traj_encoder = torch.load("/mnt/nfs2/zhanghe/project001/METRA/exp/Debug_Kitchen_original/sd000_1719919248_kitchen_metra/traj_encoder6000.pt")
    # # eval mode
    # agent_policy = load_option_policy['policy'].eval()
    # agent_traj_encoder = load_traj_encoder['traj_encoder'].eval()
    
    # open the env, and set the init lists
    frames = []
    All_Repr_obs_list = []
    All_Goal_obs_list = []
    
    Pepr_viz = True
    option_dim = 2

    # goals = [[1,1], [-1, -1], [1, -1], [-1, 1]]
    goals = [[1,-1], [-1, 1], [1, 1]]
    goals = torch.tensor(np.array(goals)).to(device)

    path_len = args.max_path_length

    for i in range(len(goals)):
        env.reset()
        obs = env.reset()  
        Repr_obs_list = []
        Repr_goal_list = []
        
        # in the kichen env
        # obs_img = env.last_state['image']
        # frames.append(obs_img)

        goal = torch.zeros(1,option_dim).to(device)
        for t in range(path_len):
            # caluculate the representaions
            obs = torch.tensor(obs).unsqueeze(0).to(device)
            # calculate the option:
            option = goals[i].unsqueeze(0)
            if t == 0:
                print("option", option)
            obs_option = torch.cat((obs, option), -1)
            # for viz
            # if Pepr_viz:
            #     phi_obs = agent_traj_encoder(obs).mean
            #     Repr_obs_list.append(phi_obs.cpu().detach().numpy()[0])
            #     Repr_goal_list.append(option.cpu().detach().numpy()[0])

            # get actions from policy
            # action = agent_policy(obs_option)[1]['mean']
            action = env.action_space.sample()

            # interact with the env
            # obs, reward, done, info = env.step(action.cpu().detach().numpy()[0])
            obs, reward, done, info = env.step(action)

            # for saving 
            obs_img = info['image']
            frames.append(obs_img)
        All_Repr_obs_list.append(Repr_obs_list)
        All_Goal_obs_list.append(Repr_goal_list)

    # # save the env as gif
    # path = "/mnt/nfs2/zhanghe/project001/METRA/tests/videos/"
    # path_file = path + "kitchen_test.gif"
    # imageio.mimsave(path_file, frames, duration=1/24)
    # print('video saved')
    # if Pepr_viz:
    #     PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=path_len)
    #     print('Repr_traj saved')



