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
        if model_name in key:
            if 'std' in key:
                data_dict[model_name+'_std'] = np.array(all_data[key])
            elif 'mean' in key:
                data_dict[model_name] = np.array(all_data[key])
            else:
                data_dict[model_name] = np.array(all_data[key])
            
    return data_dict


def plot_data(data_dict, key, label, color):
    epochs = data_dict['epoch'] * traj_batch_size * max_path_length
    interval = 1
    is_std = 0
    
    epoch_subset = epochs[epochs % interval == 0]
    data_subset = data_dict[key][epochs % interval == 0]
    if is_std == 1:
        data_subset_std = data_dict[key+'_std'][epochs % interval == 0]
    else:
        data_subset_std = 0
    
    # 高斯平滑
    # from scipy.ndimage import gaussian_filter1d
    # data_subset = gaussian_filter1d(data_subset, sigma=2)
    
    data_subset[0] = 0
    plt.plot(epoch_subset, data_subset, label=label, color=color)
    plt.fill_between(epochs, data_subset-data_subset_std, data_subset+data_subset_std, color=color, alpha=0.1)



#1. load data
plt.figure(figsize=(10, 6))
env_name = 'Large'
all_data = pd.read_csv('/mnt/nfs2/zhanghe/NuAgent/wandb_export_2024-12-23T17_29_34.313+08_00.csv', index_col=None)
# all_data = all_data.dropna(how='all', subset=['Name: dads - MjNumUniqueCoords'])
print(all_data)

#2. setings:
x_label = 'Steps'
max_path_length = 300
traj_batch_size = 16

#3. Models
# ## Ours
model_name = 'Ours' 
data_ours = pd.read_csv('/mnt/nfs2/zhanghe/NuAgent/wandb_export_2024-12-13T16_26_22.570+08_00.csv', index_col=None)
data_dict = get_data(all_data, model_name)
plot_data(data_dict, key=model_name, label='RSD', color='red')

## METRA
model_name = 'baseline' 
data_dict = get_data(all_data, model_name)
plot_data(data_dict, key=model_name, label='METRA', color='blue')

## LSD
model_name = 'LSD' 
data_dict = get_data(all_data, model_name)
plot_data(data_dict, key=model_name, label='LSD', color='green')

## DIAYN
model_name = 'DIAYN' 
data_dict = get_data(all_data, model_name)
plot_data(data_dict, key=model_name, label='DIAYN', color='grey')

## Dads
model_name = 'dads' 
data_dict = get_data(all_data, model_name)
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


