import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm


def calc_eval_metrics(trajectories, is_option_trajectories, coord_dims=[0,1], k=5):
    eval_metrics = {}

    coords = []
    for traj in trajectories:
        traj1 = traj['env_infos']['coordinates'][:, coord_dims]
        traj2 = traj['env_infos']['next_coordinates'][-1:, coord_dims]
        coords.append(traj1)
        coords.append(traj2)
    coords = np.concatenate(coords, axis=0)
    uniq_coords = np.unique(np.floor(k * coords), axis=0)
    eval_metrics.update({
        'MjNumUniqueCoords': len(uniq_coords),
    })

    return eval_metrics


# save the traj. as fig
def PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=100, is_PCA=False, is_goal=True):
    if len(All_Goal_obs_list) == 0:
        is_goal = False
    
    Repr_obs_array = np.array(All_Repr_obs_list[0])
    if is_goal:
        All_Goal_obs_array = np.array(All_Goal_obs_list[0])
    for i in range(1,len(All_Repr_obs_list)):
        Repr_obs_array = np.concatenate((Repr_obs_array, np.array(All_Repr_obs_list[i])), axis=0)
        if is_goal:
            All_Goal_obs_array = np.concatenate((All_Goal_obs_array, np.array(All_Goal_obs_list[i])), axis=0)
    # 创建 PCA 对象，指定降到2维
    if is_PCA:
        pca = PCA(n_components=2)
        # 对数据进行 PCA
        Repr_obs_2d = pca.fit_transform(Repr_obs_array)
    else:
        Repr_obs_2d = Repr_obs_array
        if is_goal:
            All_Goal_obs_2d = All_Goal_obs_array
    # 绘制 PCA 降维后的数据
    plt.figure(figsize=(8, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(All_Repr_obs_list)))
    for i in range(0,len(All_Repr_obs_list)):
        color = colors[i]
        start_index = i * path_len
        end_index = (i+1) * path_len
        plt.scatter(Repr_obs_2d[start_index:end_index, 0], Repr_obs_2d[start_index:end_index, 1], color=color, s=5)
        if is_goal:
            plt.scatter(All_Goal_obs_2d[start_index:end_index, 0], All_Goal_obs_2d[start_index:end_index, 1], color=color, s=100, marker='*', edgecolors='black')
    path_file_traj = path + "-traj.png"
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('traj. in representation space')
    # plt.legend()
    plt.savefig(path_file_traj)
    
    
def viz_SZN_dist(SZN, input_token, path):
    dist = SZN(input_token)
    # Data
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x,y)
    from scipy.stats import multivariate_normal
    num = dist.mean.shape[0]
    fig = plt.figure(figsize=(18, 12), facecolor='w')
    for i in range(dist.mean.shape[0]):
        # Multivariate Normal
        mu_x = dist.mean[i][0].detach().cpu().numpy()
        sigma_x = dist.stddev[i][0].detach().cpu().numpy()
        mu_y = dist.mean[i][1].detach().cpu().numpy()
        sigma_y = dist.stddev[i][1].detach().cpu().numpy()
        rv = multivariate_normal([mu_x, mu_y], [[sigma_x, 0], [0, sigma_y]])
        # Probability Density
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        pd = rv.pdf(pos)
        # Plot
        ax = fig.add_subplot(2, num//2, i+1, projection='3d')
        ax.plot_surface(X, Y, pd, cmap='viridis', linewidth=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')
        ax.set_title(label = str(mu_x)[:5] + '-' + str(sigma_x)[:4] + '\n' + str(mu_y)[:5] + '-' + str(sigma_y)[:4])
    plt.savefig(path + '-all' + '.png')
    plt.close()
