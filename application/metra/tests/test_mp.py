import multiprocessing as mp
import gym


def get_buffer_map(env):
    buffer = []
    env.reset()
    for _ in range(10):
        obs, action, reward, info = env.step(env.action_space.sample())
        buffer.append((obs, action, reward, info))

    return buffer

def foo(q, message, i, env):

    buffer = get_buffer_map(env)

    q.put(buffer)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    processes = []
    Buffer = []
    env = gym.make('CartPole-v0')
    
    # 创建并启动多个进程
    for i in range(5):
        # 创建进程, traget是进程执行的函数, args是传递给函数的参数
        p = mp.Process(target=foo, args=(q, f'hello from process {i}', i, env))
        p.start()
        processes.append(p)
    
    # 从队列中获取每个进程放入的数据
    for _ in range(5):
        Buffer.append(q.get())
    
    # 等待所有进程完成
    for p in processes:
        p.join()

    print(Buffer)




    
    