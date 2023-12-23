import time

from argparse import Namespace
from collections import deque
from itertools import count

import torch

from torch import nn
from torch.nn import functional as F

from tensorboardX import SummaryWriter

from uniagent.envs.atari import create_atari_env
from uniagent.envs.gym_control import create_gym_control
from application.a3c_gym.agent import Agent


def rollout(epoch, counter, lock, agent, shared_model, env, start_time, writer, args):
    obs, info = env.reset()
    obs = torch.from_numpy(obs).to(args.device)
    done = True
    actions = deque(maxlen=100)
    reward_sum = 0

    agent.model.eval()
    for episode_length in count():
        if done:
            net_state = agent.init_state(1, args.device)
        else:
            net_state = tuple(map(lambda x: x.detach(), net_state))

        with torch.no_grad():
            value, action, log_prob, entropy, net_state = agent.act(
                obs.unsqueeze(0), net_state
            )
            action = action.cpu().numpy()[0]

        obs, reward, done, truncated, info = env.step(action)

        done = done or truncated
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action)
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print(
                "Time {}, epoch {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    epoch,
                    counter.value,
                    counter.value / (time.time() - start_time),
                    reward_sum,
                    episode_length,
                )
            )
            with lock:
                for name, param in shared_model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                writer.add_scalar(
                    "evaluation/episode_reward", reward_sum, counter.value
                )

            actions.clear()
            break

        obs = torch.from_numpy(obs).to(args.device)


def test(
    rank: int, shared_model: nn.Module, counter, log_dir: str, lock, args: Namespace
):
    torch.manual_seed(args.seed + rank)

    if args.task_type == "atari":
        env = create_atari_env(
            args.env_name, args.max_episode_length, use_reward_clip=False
        )
    elif args.task_type == "control":
        env = create_gym_control(args.env_name)

    if hasattr(env, "seed"):
        env.seed(args.seed + rank)

    agent = Agent(
        shared_model.__class__,
        env.observation_space,
        env.action_space,
        args.device,
    )
    if not args.async_mode:
        agent.model = shared_model

    writer = SummaryWriter(log_dir=log_dir)

    agent.model.eval()
    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    for epoch in count():
        # Sync with the shared model
        if args.async_mode:
            agent.model.load_state_dict(shared_model.state_dict())

        done = rollout(
            epoch,
            counter,
            lock,
            rank,
            agent,
            shared_model,
            env,
            start_time,
            writer,
            args,
        )
        time.sleep(10)
