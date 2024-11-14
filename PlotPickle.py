import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

def PlotCurve(data, key, path, ax, label):
    x = data['epoch']
    y = data[key]
    ax.plot(x, y, label=label)
    ax.set_title(key)
    # plt.legend()
    

def LoadDictData(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    for dict_i in data:
        if dict_i['epoch'] == '0':
            dict_data = {}
            dict_data = defaultdict(list) 
        for k, v in dict_i.items():
            dict_data[k].append(float(v))
    return dict_data


def PlotSaveKey(key, paths, labels):
    fig, ax = plt.subplots()
    for i in range(len(paths)):
        dict_data = LoadDictData(paths[i])
        PlotCurve(dict_data, key, path=paths[i], ax=ax, label=labels[i])
        
    filepath = str(key) + '.png'
    plt.legend()
    plt.savefig(filepath)
    print(f'saved as {filepath}')


path1 = '/mnt/nfs2/zhanghe/NuAgent/exp/LittleMaze/baselinesd000_1731046271_lm_PSZP/metrics_random.csv'
path2 = '/mnt/nfs2/zhanghe/NuAgent/exp/LittleMaze/SZN-t05-no_g_dirsd000_1731067494_lm_PSZP/metrics_random_psi.csv'

paths = [path1, path2]
labels = ["metra", "ours"]
# PlotSaveKey(key='FD')
# PlotSaveKey(key='AR')
PlotSaveKey(key='CoverCoords', paths=paths, labels=labels)

