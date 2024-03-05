from typing import Any
import time

from argparse import Namespace
from collections import deque
from itertools import count

import torch

from torch import nn
from torch.nn import functional as F

from tensorboardX import SummaryWriter
from uniagent.envs.atari import create_atari_env
from .policy import Agent


def rollout(epoch, counter, agent: Agent, env, start_time, writer, args):
    obs, info = env.reset()
    obs = torch.from_numpy(obs).to(args.device)
    actions = deque(maxlen=100)
    reward_sum = 0
    done = False
    net_states = agent.init_state(1, args.device)

    for step_cnt in count():
        with torch.no_grad():
            action, net_states = agent.act(obs.unsqueeze(0), net_states)
        obs, reward, done, truncated, info = env.step(action.item())
        done = done or truncated
        reward_sum += reward

        actions.append(action.item())
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
                    step_cnt,
                )
            )
            writer.add_scalar("eval/episode_reward", reward_sum, counter.value)
            writer.add_scalar("eval/episode_length", step_cnt, counter.value)

            actions.clear()
            break
        obs = torch.from_numpy(obs).to(args.device)
    return done


def test(
    rank: int,
    env: Any,
    shared_model: nn.Module,
    counter,
    log_dir: str,
    lock,
    args: Namespace,
    model_cls: nn.Module,
):
    torch.manual_seed(args.seed + rank)

    # env = create_atari_env(
    #     args.env_name, args.max_episode_length, use_reward_clip=False
    # )
    if hasattr(env, "seed"):
        env.seed(args.seed + rank)

    if args.async_mode:
        agent = Agent(model_cls, env.observation_space, env.action_space, args.device)
    else:
        del agent.model
        agent.model = shared_model

    writer = SummaryWriter(log_dir=log_dir)

    agent.model.eval()
    start_time = time.time()
    done = True

    for epoch in count():
        # Sync with the shared model
        if args.async_mode:
            agent.model.load_state_dict(shared_model.state_dict())

        done = rollout(epoch, counter, agent, env, start_time, writer, args)
        time.sleep(5)
