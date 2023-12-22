import time

from argparse import Namespace
from collections import deque
from itertools import count

import torch

from torch import nn

from uniagent.models.fun import FeudalNet
from uniagent.envs.atari import create_atari_env

from tensorboardX import SummaryWriter


def test(
    rank: int, shared_model: nn.Module, counter, log_dir: str, lock, args: Namespace
):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(
        args.env_name, args.max_episode_length, use_normalized_env=False
    )
    env.seed(args.seed + rank)
    model = FeudalNet(
        env.observation_space,
        env.action_space,
        channel_first=args.channel_first,
        device=args.device,
    )

    writer = SummaryWriter(log_dir=log_dir)

    model.eval()

    obs, info = env.reset()
    obs = torch.from_numpy(obs).to(args.device)
    done = True
    reward_sum = 0

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    for epoch in count():
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        states = model.init_state(1)
        # else:
        #     states = model.reset_states_grad(states)

        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(args.device)
        done = False

        for episode_length in count():
            value_worker, value_manager, action_probs, goal, _, states = model(
                obs.unsqueeze(0), states, reset_value_grad=True
            )
            action = action_probs.cpu().max(1, keepdim=True)[1].data.numpy()

            obs, reward, done, truncated, _ = env.step(action[0, 0])
            done = done or truncated
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            actions.append(action[0, 0])
            if actions.count(actions[0]) == actions.maxlen:
                done = True

            if done:
                print(
                    "Time {}, epoch {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                        time.strftime(
                            "%Hh %Mm %Ss", time.gmtime(time.time() - start_time)
                        ),
                        epoch,
                        counter.value,
                        counter.value / (time.time() - start_time),
                        reward_sum,
                        episode_length,
                    )
                )
                with lock:
                    for name, param in shared_model.named_parameters():
                        writer.add_histogram(
                            name, param.clone().cpu().data.numpy(), epoch
                        )
                    writer.add_scalar(
                        "evaluation/episode_reward", reward_sum, counter.value
                    )

                reward_sum = 0
                actions.clear()
                obs, info = env.reset()
                time.sleep(30)
                break

            obs = torch.from_numpy(obs).to(args.device)
