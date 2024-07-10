import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio

from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze, plot_trajectories
import d4rl

# setup the env
env = GoalReachingMaze("antmaze-medium-diverse-v0")

env.reset()

frames = []

fig, ax = plt.subplots()

# goal 
# goals = env.four_goals()
# 随机Sample数据：
np_random = np.random.default_rng() 
goals = env.env.goal_sampler(np_random)
print(goals)

ax.scatter(goals[0], goals[1], s=50, marker='*', alpha=1, edgecolors='black')
env.draw(ax)


trajs = []
for i in range(2):
    env.reset()
    traj_list = {}
    traj_list["observation"] = []
    traj_list["info"] = []
    for t in range(100):
        action =  env.action_space.sample()
        obs, reward, dones, info = env.step(action)
        print(info)
        
        traj_list["observation"].append(obs["observation"])
        traj_list["info"].append(info)
        
        frames.append(env.render(mode='rgb_array'))
        
    trajs.append(traj_list)
    
# save the env as gif
# path = "/data/zh/project12_Metra/METRA/tests/videos/"
# path_file = path + "antmaze_test.gif"
# imageio.mimsave(path_file, frames, duration=1/24)
# print('video saved')

# plot_trajectories 用来绘制轨迹；
# print(trajs)
plot_trajectories(env, trajs, fig, ax)
# 保存图片；
path = "/data/zh/project12_Metra/METRA/tests/videos/"
plt.savefig(path + "maze.png")
