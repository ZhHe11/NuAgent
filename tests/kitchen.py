import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
from tqdm import trange

env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode='rgb_array')
# 重置环境
obs = env.reset()

# 打开 HDF5 文件
import h5py
with h5py.File('/home/zhangh/.d4rl/datasets/mini_kitchen_microwave_kettle_light_slider-v0.hdf5', 'r') as f:
    # 列出文件中的所有组
    print("Keys: %s" % f.keys())
    
    # 选择一个数据集
    obs = f['observations'][:]
    action = f['actions'][:]
    infos = f['infos'][:]
    terminals = f['terminals'][:]
    timeouts = f['timeouts'][:]

count = 0
obs = env.reset()
frames = []
for i in trange(len(action)):
    if terminals[i] == False and timeouts[i] == False:   
        a = action[i]
        env.step(a)
        frames.append(env.render())
        
    else:
        a = action[i]
        env.step(a)
        frames.append(env.render())
        
        gif_name = '/data/zh/project12_Metra/METRA/tests/videos/FrankaKitchen-v1' + str(count) + '.gif'
        imageio.mimsave(gif_name, frames, 'GIF', duration=1)
        print('saved')
        
        frames = []
        count += 1
        obs = env.reset()



# frames = []
# for i in trange(100):
#     action = env.action_space.sample()
#     env.step(action)
#     frames.append(env.render())
    
# gif_name = '/data/zh/project12_Metra/METRA/tests/videos/FrankaKitchen-v1'+ '.gif'
# imageio.mimsave(gif_name, frames, 'GIF', duration=1)
# print('saved')
    
