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
            if 'MIN' in key:
                data_dict[model_name+'_min'] = np.array(all_data[key])
            elif 'MAX' in key:
                data_dict[model_name+'_max'] = np.array(all_data[key])
            else:
                data_dict[model_name] = np.array(all_data[key])
                                
    return data_dict


def plot_data(data_dict, key, label, color):
    epochs = data_dict['epoch'] * traj_batch_size * max_path_length
    plt.plot(epochs, data_dict[key], label=label, color=color)
    plt.fill_between(epochs, data_dict[key+'_min'], data_dict[key+'_max'], color=color, alpha=0.2)



#1. load data
env_name = 'Ant'
plt.figure(figsize=(10, 6))
all_data = pd.read_csv('/mnt/nfs2/zhanghe/NuAgent/wandb_export_2024-12-11T12_07_43.828+08_00.csv', index_col=None)

all_data = all_data.dropna(how='all', subset=['Name: dads - MjNumUniqueCoords'])

#2. setings:
x_label = 'Steps'
max_path_length = 200
traj_batch_size = 16

#3. Models
## Ours
model_name = 'Ours' 
data_dict = get_data(all_data, model_name)
plot_data(data_dict, key=model_name, label='PDSD', color='red')

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


