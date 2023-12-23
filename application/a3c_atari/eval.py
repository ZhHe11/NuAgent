import time

from argparse import Namespace
from collections import deque
from itertools import count

import torch

from torch import nn
from torch.nn import functional as F

from uniagent.models.a2c import ActorCritic
from uniagent.envs.atari import create_atari_env

from tensorboardX import SummaryWriter


def rollout(
    epoch, counter, lock, rank, model, shared_model, env, start_time, writer, args
):
    obs, info = env.reset()
    obs = torch.from_numpy(obs).to(args.device)
    done = True
    actions = deque(maxlen=100)
    reward_sum = 0

    for episode_length in count():
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256).to(args.device)
            hx = torch.zeros(1, 256).to(args.device)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((obs.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.cpu().max(1, keepdim=True)[1].numpy()

        obs, reward, done, truncated, info = env.step(action[0, 0])

        done = done or truncated
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
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

    env = create_atari_env(
        args.env_name, args.max_episode_length, use_reward_clip=False
    )
    env.seed(args.seed + rank)

    if args.async_mode:
        model = ActorCritic(
            env.observation_space.shape[0],
            env.action_space,
        )
        model.to(args.device)
    else:
        model = shared_model

    writer = SummaryWriter(log_dir=log_dir)

    model.eval()
    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    for epoch in count():
        # Sync with the shared model
        if args.async_mode:
            model.load_state_dict(shared_model.state_dict())

        done = rollout(
            epoch,
            counter,
            lock,
            rank,
            model,
            shared_model,
            env,
            start_time,
            writer,
            args,
        )
        time.sleep(10)
