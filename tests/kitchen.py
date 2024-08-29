import gym
import d4rl
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('kitchen-complete-v0')

# 重置环境
obs = env.reset()

for i in range(100):
    action = env.action_space.sample()
    env.step(action)
    env.render(mode='human')
    
    
    
