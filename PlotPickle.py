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



path2 = '/mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/baselinesd000_1733132528_lm_metra_bl/metrics_random.csv'

path3 = '/mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/baselinesd000_1733133625_lm_metra_bl/metrics_random.csv'

path4 = '/mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/baselinesd002_1733235429_lm_metra_bl/metrics_random.csv'

path5 = '/mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/baselinesd004_1733241785_lm_metra_bl/metrics_random.csv'

# path2 = '/mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/Ours-OnlyRegretScale-lr_te_1e_3-wodsd000_1733208892_lm_SZPC/metrics_random_psi.csv'

# path_2 = ''



labels = ["metra", "ours"]
# PlotSaveKey(key='FD')
# PlotSaveKey(key='AR')
# PlotSaveKey(key='CoverCoords', paths=paths, labels=labels)

import numpy as np
import matplotlib.pyplot as plt



def GetXY(paths, x_key, y_key):
    x = []
    y = []
    for i in range(len(paths)):
        dict_i = LoadDictData(paths[i])
        x.append(dict_i[x_key])
        y.append(dict_i[y_key])
        
    return x, y



def PlotCurve(x,y,ModelName,color):
    # 计算每个时间步的均值和标准差
    x = np.array(x)
    y = np.array(y)
    mean_rewards = np.mean(y, axis=0)
    std_rewards = np.std(y, axis=0)
    
    # 绘制均值曲线
    plt.plot(x[0], mean_rewards, label=ModelName, color=color)

    # 添加标准差阴影
    plt.fill_between(x[0],
                    mean_rewards - std_rewards,
                    mean_rewards + std_rewards,
                    color=color, alpha=0.2)



# load data
x_key = 'epoch'
y_key = 'CoverCoords'

plt.figure(figsize=(10, 6))

paths = [path2, path3, path4, path5]
x,y = GetXY(paths, x_key, y_key)
PlotCurve(x,y,ModelName='METRA',color='blue')

paths = [
            '/mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/Ours-OnlyRegretScale-lr_te_1e_3-wodsd000_1733208892_lm_SZPC/metrics_random_psi.csv',
            '/mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/Ours-OnlyRegretScale-lr_te_1e_3-wodsd004_1733213229_lm_SZPC/metrics_random_psi.csv',
            '/mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/Ours-OnlyRegretScale-lr_te_1e_3-wodsd008_1733213261_lm_SZPC/metrics_random_psi.csv',
        ]
x,y = GetXY(paths, x_key, y_key)
PlotCurve(x,y,ModelName='PDSD',color='red')


# 图表美化
plt.xlabel('Epoch')
plt.ylabel('CoverCoords')
plt.title('Little Maze Large')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()
plt.savefig('LittleMazeLarge')
print('save as ./LittleMazeLarge.png')


