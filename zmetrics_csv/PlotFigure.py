import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import pandas as pd


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_data(all_data, key):
    data_dict = {}
    data_dict['epoch'] = np.array(all_data['epoch'])
    for key in all_data.columns:
        data_dict[key] = np.array(all_data[key])
    
    return data_dict


def plot_data(data_dict, key, label, color):
    epochs = data_dict['epoch'] * traj_batch_size * max_path_length
    CoverCoords1 = data_dict['CoverCoords1']
    CoverCoords2 = data_dict['CoverCoords2']
    CoverCoords3 = data_dict['CoverCoords3']
    
    data_stack = np.stack([CoverCoords1, CoverCoords2, CoverCoords3])
    data_mean = data_stack.mean(axis=0)
    data_std = data_stack.std(axis=0)
    
    # 高斯平滑
    # from scipy.ndimage import gaussian_filter1d
    # data_subset = gaussian_filter1d(data_subset, sigma=2)
    
    epochs = np.insert(epochs, 0, 0)
    data_mean = np.insert(data_mean, 0, 0)
    data_std = np.insert(data_std, 0, 0)
    plt.plot(epochs, data_mean, label=label, color=color)
    plt.fill_between(epochs, data_mean-data_std, data_mean+data_std, color=color, alpha=0.1)
    
    plt.savefig('./test.png')

#1. load data
plt.figure(figsize=(10, 6))
env_name = 'AntMaze'

#2. setings:
x_label = 'Steps'
max_path_length = 300
traj_batch_size = 16

#3. Models
# ## Ours
# model_name = 'Ours' 
# data_ours = pd.read_csv('/mnt/nfs2/zhanghe/NuAgent/zmetrics_csv/AntMaze/AntMazeMetricsOurs.csv', index_col=None)
# data_dict = get_data(all_data, model_name)
# plot_data(data_ours, key=model_name, label='PDSD', color='red')

## METRA
model_name = 'baseline' 
data = pd.read_csv('/mnt/nfs2/zhanghe/NuAgent/zmetrics_csv/AntMaze/AntMazeMetricsBaselineE100.csv', index_col=None)
data_dict = get_data(data, model_name)
plot_data(data_dict, key=model_name, label='METRA', color='blue')

# ## LSD
model_name = 'LSD' 
data = pd.read_csv('/mnt/nfs2/zhanghe/NuAgent/zmetrics_csv/AntMaze/AntMazeMetricsLSDE100.csv', index_col=None)
data_dict = get_data(data, model_name)
plot_data(data_dict, key=model_name, label='LSD', color='green')

# ## DIAYN
model_name = 'DIAYN' 
data = pd.read_csv('/mnt/nfs2/zhanghe/NuAgent/zmetrics_csv/AntMaze/AntMazeMetricsDIAYNE100.csv', index_col=None)
data_dict = get_data(data, model_name)
plot_data(data_dict, key=model_name, label='DIAYN', color='grey')

# ## Dads
model_name = 'dads' 
data = pd.read_csv('/mnt/nfs2/zhanghe/NuAgent/zmetrics_csv/AntMaze/AntMazeMetricsDADSE100.csv', index_col=None)
data_dict = get_data(data, model_name)
plot_data(data_dict, key=model_name, label='DADS', color='orange')


#4. Plot
plt.xlabel(x_label)
plt.ylabel('CoverCoords')
plt.title(env_name)
plt.legend()
plt.grid(True)

savepath = '/mnt/nfs2/zhanghe/NuAgent/plots/' + env_name + '.png'
plt.savefig(savepath)
print(f'saved as {savepath}')




