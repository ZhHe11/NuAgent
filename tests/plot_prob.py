from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx, GaussianMLPIndependentStdModuleEx, GaussianMLPModuleEx, XY_GaussianMLPIndependentStdModuleEx
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def get_gaussian_module_construction(args,
                                     *,
                                     hidden_sizes,
                                     const_std=False,
                                     hidden_nonlinearity=torch.relu,
                                     w_init=torch.nn.init.xavier_uniform_,
                                     init_std=1.0,  # 1.0
                                     min_std=1e-6,  # 1e-6
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

import os 
import sys
sys.path.append('/data/zh/project12_Metra/METRA/')
from GetArgparser import get_argparser
args = get_argparser().parse_args()

master_dims = [args.model_master_dim] * args.model_master_num_layers

# Network for goal policy
# zhanghe
module_cls, module_kwargs = get_gaussian_module_construction(
    args,
    hidden_sizes=master_dims,
    hidden_nonlinearity=torch.relu,
    w_init=torch.nn.init.xavier_uniform_,
    input_dim=args.dim_option,
    output_dim=args.dim_option,
    init_std=1.0,
    min_std=1e-6,
    max_std=1e6,
)
goal_sample_network = module_cls(**module_kwargs).to('cuda')

def vec_norm(vec):
    return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)

SampleGoalNet = torch.load("/data/zh/project12_Metra/METRA/SampleGoalNet.pt")['SampleGoalNet']


directions = vec_norm(torch.randn((100, 2))).to('cuda')

dist = SampleGoalNet(directions)

mean = dist.mean.detach()
stddev = dist.stddev.detach()

edge_mean = (directions * mean).cpu().numpy()

edge_std = (directions * (mean+stddev)).cpu().numpy()


plt.figure(figsize=(8, 8))
plt.scatter(x=edge_mean[:,0], y=edge_mean[:,1])
plt.scatter(x=edge_std[:,0], y=edge_std[:,1])
# plt.colorbar(label='Probability Density')
plt.title('Edge')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('test.png')


# # 生成网格数据
# x, y = np.mgrid[-5:5:.01, -5:5:.01]
# pos = np.dstack((x, y))

# total_density = np.zeros(pos.shape[0:2])

# tensor_x = torch.tensor(x, dtype=torch.float32)
# tensor_y = torch.tensor(y, dtype=torch.float32)
# tensor_pos = torch.tensor(pos, dtype=torch.float32).to('cuda')
# s_0 = torch.zeros((1,2)).to('cuda')
# with torch.no_grad():
#     dist = goal_sample_network(s_0)

# # prob = torch.exp(dist.log_prob(tensor_pos))
# prob = dist.log_prob(tensor_pos)


# prob = prob.cpu().numpy()
# plt.figure(figsize=(8, 8))
# plt.imshow(prob, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis', interpolation='nearest')
# plt.colorbar(label='Probability Density')
# plt.title('Probability Density Heatmap')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.savefig('test.png')











