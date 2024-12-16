import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import torch
import torch.distributions as dist

def UpdateGMM(dists, GMM=None, mix_dist_prob=None, device='cuda'):
    if GMM is None:
        component_distribution = dist.Independent(
            dist.Normal(
                loc=torch.stack([g.mean[0] for g in dists]),
                scale=torch.stack([g.stddev[0] for g in dists])
            ),
            reinterpreted_batch_ndims=1
        )
        if mix_dist_prob is None:
            # 创建均匀的 mixture_distribution
            mixture_distribution = dist.Categorical(
                probs=(torch.ones(len(dists)) / len(dists)).to(device)
            )
        else: 
            mixture_distribution = dist.Categorical(
                probs=mix_dist_prob
            )
        # 组合成一个 MixtureSameFamily 分布
        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )
        return window_dist

    else:
        component_distribution = GMM.component_distribution
        mixture_distribution = mixture_distribution

        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )

        return window_dist




SZN_load = torch.load('/mnt/nfs2/zhanghe/NuAgent/exp/Large/woAdp-trianMoresd000_1734058195_ant_maze_large_SZPC3/wandb/latest-run/filesSampleZPolicy-3000.pt')

window = SZN_load['window']
device = 'cuda'
window_dist = UpdateGMM(window, mix_dist_prob=None, device=device)

x_grid = np.linspace(-1, 1, 100)
y_grid = np.linspace(-1, 1, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T



grid_points_tensor = torch.tensor(grid_points).to(device)
zeros_tensor = torch.zeros(grid_points_tensor.shape[0], 2).to(device)
grid_points_tensor_expanded = torch.cat((grid_points_tensor, zeros_tensor), dim=1)

log_prob = window_dist.log_prob(grid_points_tensor_expanded).cpu().numpy()
prob_density = np.exp(log_prob).reshape(X_grid.shape)

plt.figure(figsize=(8, 6))
plt.contourf(X_grid, Y_grid, prob_density, 20, cmap='viridis')
# plt.scatter(x, y, s=5, color='red', alpha=0.5)
plt.title('GMM 4D (projected to 2D) Probability Density')
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar(label='Density')
plt.savefig('GMM.png')

