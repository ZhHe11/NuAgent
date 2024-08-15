import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm


def PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=100, is_PCA=False, tag=''):
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
    file_path = os.path.join(path, tag + "repr_traj.png")
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('traj. in representation space')
    plt.legend()
    plt.savefig(file_path)
    wandb.log(({ tag + "Repr_Space_traj" : wandb.Image(file_path)}))


