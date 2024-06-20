import gym
import argparse
import os
import imageio

# 解析命令行参数
parser = argparse.ArgumentParser(description='Gym environment video recorder')
parser.add_argument('--render', default=1, action='store_true', help='Render and record the environment')
parser.add_argument('--save_dir', type=str, default='/home/PJLAB/zhanghe/Projects/PJ004-METRA/myMetra/universal-agent/application/metra/tests/video', help='Directory to save videos')
parser.add_argument('--env_name', type=str, default='Ant-v4', help='Name of the environment')
args = parser.parse_args()

# # 确保保存目录存在
# os.makedirs(args.save_dir, exist_ok=True)

# # 创建环境
# env = gym.make(args.env_name, render_mode='human')
env = gym.make(args.env_name, render_mode='rgb_array')

def save_frames_as_gif(frames, filename):
    path = "/home/PJLAB/zhanghe/Projects/PJ004-METRA/myMetra/universal-agent/application/metra/tests/video"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, filename)
    imageio.mimsave(filename, frames, duration=1/10)
    
# 运行一个回合
for i in range(10):
    print(f'Running episode {i+1}')
    # 初始化环境
    obs = env.reset()
    done = False
    total_reward = 0
    
    timestep = 0

    # 保存帧
    frames = []


    while not done:
        timestep += 1
        if timestep > 100:
            break
        # 环境渲染（可选）
        if args.render:
            frame = env.render()
            print(frame)
            frames.append(frame)
        
        # 随机选择一个动作  
        action = env.action_space.sample()
        
        # 应用动作，获取下一个状态
        obs, reward, done, _, info = env.step(action)
        # total_reward += reward
    
    save_frames_as_gif(frames=frames, filename=args.env_name + str(i) + '.gif')

# 打印总奖励
print(f'Total reward: {total_reward}')

# 关闭环境
env.close()