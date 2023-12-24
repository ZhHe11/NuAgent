from argparse import Namespace
from itertools import count

import gym
import torch
import torch.distributed.rpc as rpc

from uniagent.envs.atari import create_atari_env
from uniagent.envs.gym_control import create_gym_control


class Observer:
    def train(self, args, agent_rref, act_fn, value_fn):
        if args.task_type == "atari":
            env = create_atari_env(args.env_name)
        elif args.task_type == "control":
            env = create_gym_control(args.env_name)

        task_id = rpc.get_worker_info().id - 1

        for epoch in count():
            rewards = []
            discounted_rets = []
            values = []
            log_probs = []
            entropies = []

            obs, info = env.reset()

            for step in range(args.num_steps):
                obs = torch.from_numpy(obs).float()
                value, action, log_prob, entropy = rpc.rpc_sync(
                    agent_rref.owner(), act_fn, args=(agent_rref, task_id, obs)
                )
                obs, reward, done, truncated, info = env.step(action)

                done = done or truncated

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropies.append(entropy)

                if done:
                    break

            if not done:
                value = rpc.rpc_sync(
                    agent_rref.owner(), value_fn, args=(agent_rref, task_id, obs)
                )
                values.append(value.squeeze())
            else:
                values.append(torch.zeros(1).to(args.device).squeeze())

            R = values[-1]
            traj_len = len(rewards)
            discounted_rets = [0.0] * traj_len
            for i in range(traj_len - 1, -1, -1):
                R = rewards[i] + args.gamma * R
                discounted_rets[i] = R
                delta_t = (
                    rewards[i]
                    + args.gamma * values[i + 1].cpu().item()
                    - values[i].cpu().item()
                )

                gae = gae * args.gamma * args.llambda + delta_t
                adv = discounted_rets[i] - values[i]

                value_loss = 0.5 * adv.pow(2)

            # TODO(ming): compute and loss train here
