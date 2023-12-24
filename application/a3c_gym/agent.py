from typing import Tuple
from gym import spaces

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from uniagent.models.a2c import ActorCritic
from application.dqn_gym.policy import AtariPreprocessor


class AtariAC(ActorCritic):
    def create_preprocessor(self, num_outputs: int) -> nn.Module:
        return AtariPreprocessor(self.observation_space, num_outputs)


class Agent:
    def __init__(
        self,
        model_class: nn.Module,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.DeviceObjType = None,
    ) -> None:
        self.device = device
        self.action_space = action_space
        self.model = model_class(observation_space, action_space).to(device)

    def act(
        self, obs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        value, logits, (hx, cx) = self.model(obs, states)

        dist = Categorical(logits=logits)
        if self.model.training:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return value, action, log_prob, entropy, (hx, cx)

    def init_state(self, batch_size: int, device=None):
        states = self.model.init_state(batch_size, device)
        return states


from typing import Any

import time
import threading

from itertools import count
from argparse import Namespace

import numpy as np
import torch.distributed.rpc as rpc

from torch.distributed.rpc import RRef
from tensorboardX import SummaryWriter

from uniagent.trainers.parameter_server import ParameterServer


class AsyncAgent:
    def __init__(
        self,
        model_class: nn.Module,
        model_kwargs: dict,
        make_env: Any,
        device: torch.DeviceObjType = None,
    ) -> None:
        self.model = model_class(**model_kwargs).to(device)
        self.device = device
        self.env = make_env()
        self.worker_name = rpc.get_worker_info().name
        self.lock = threading.Lock()

    def run_episode(
        self, args: Namespace, model: nn.Module, global_counter=None, lock=None
    ):
        obs, _ = self.env.reset()
        done = False

        rewards = []
        values = []
        log_probs = []
        entropies = []
        obses = []
        actions = []
        net_states = []

        # make sure it is not None
        net_state = model.init_state(1, self.device)

        counter = count() if model.training else range(args.num_steps)

        for step_cnt in counter:
            obses.append(obs)
            net_states.append([e.cpu().numpy().squeeze(0) for e in net_state])
            obs = torch.from_numpy(obs).float()
            value, logits, net_state = self.model(obs.unsqueeze(0), net_state)

            dist = Categorical(logits=logits)
            if self.model.training:
                action = dist.sample()
            else:
                action = logits.argmax(dim=-1)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            obs, reward, done, truncated, info = self.env.step(action.cpu().numpy()[0])
            done = done or truncated

            values.append(value.squeeze())
            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.squeeze())
            entropies.append(entropy.squeeze())
            rewards.append(reward)

            if global_counter:
                with lock:
                    global_counter.value += 1

            if done:
                break

        obses.append(obs)
        net_states.append(net_state)

        if done:
            values.append(torch.zeros(1).to(self.device).squeeze())
        else:
            obs = torch.from_numpy(obs).float()
            value, _, _, _, _ = self.act(obs.unsqueeze(0), net_state)
            values.append(value.squeeze())

        return (
            obses,
            actions,
            net_states,
            rewards,
            values,
            log_probs,
            entropies,
            len(rewards),
        )

    def test(self, args: Namespace, ps_rref, counter, lock, log_dir):
        start_time = time.time()
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in count():
            model: nn.Module = ps_rref.rpc_sync().get_model().to(self.device)
            model.eval()
            _, _, _, rewards, _, _, _, episode_len = self.run_episode(args, model)
            reward_sum = sum(rewards)
            print(
                "Time {}, epoch {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    epoch,
                    counter.value,
                    counter.value / (time.time() - start_time),
                    reward_sum,
                    episode_len,
                )
            )
            with lock:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                writer.add_scalar(
                    "evaluation/episode_reward", reward_sum, counter.value
                )
            time.sleep(5)

    def train(self, args: Namespace, ps_rref, counter, lock, rank, log_dir):
        model: nn.Module = ps_rref.rpc_sync().get_model().to(self.device)
        time.sleep(1)
        writer = SummaryWriter(log_dir=log_dir)

        model.train()
        for epoch in count():
            (
                obses,
                actions,
                net_states,
                rewards,
                values,
                log_probs,
                entropies,
                episode_len,
            ) = self.run_episode(args, model, counter, lock)

            gae = 0.0
            ret = values[-1].item()
            R = [0.0] * episode_len
            GAE = [0.0] * episode_len

            with lock:
                writer.add_scalars(
                    "training/episode_info" + str(rank),
                    {
                        "episode_reward": sum(rewards),
                        "episode_length": episode_len,
                    },
                    epoch,
                )
                # print(
                #     f"[training] rank-{rank}: episode_reward ({sum(rewards)}), episode_len ({episode_len})"
                # )

            for i in reversed(range(episode_len)):
                ret = args.gamma * ret + rewards[i]
                delta_t = (
                    rewards[i]
                    + args.gamma * values[i + 1].cpu().item()
                    - values[i].cpu().item()
                )
                gae = gae * args.gamma * args.llambda + delta_t

                GAE[i] = gae
                R[i] = ret

            R = torch.from_numpy(np.asarray(R)).float().to(args.device)
            GAE = torch.from_numpy(np.asarray(GAE)).float().to(args.device)
            obses = torch.from_numpy(np.stack(obses)).float().to(args.device)
            actions = (
                torch.from_numpy(np.stack(actions)).squeeze(1).long().to(args.device)
            )
            net_states = [
                torch.from_numpy(np.stack(e)).float().to(args.device)
                for e in net_states
            ]
            values, logits, _ = model(obses, net_states)
            values = values[:-1].squeeze()
            logits = logits[:-1]
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()

            assert values.shape == R.shape, (
                values.shape,
                R.shape,
            )

            Advs = R - values

            assert Advs.requires_grad
            value_loss = 0.5 * Advs.pow(2).mean()

            assert log_probs.shape == GAE.shape, (
                log_probs.shape,
                GAE.shape,
            )
            assert log_probs.requires_grad

            pg_loss = -(log_probs * GAE.detach()).mean()
            entropy_loss = entropies.mean()

            total_loss = (
                pg_loss
                + args.value_loss_coef * value_loss
                - args.entropy_coef * entropy_loss
            )

            assert total_loss.requires_grad

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )

            continue_training = True
            for p in model.parameters():
                if p.grad is None:
                    print("[WARNING] no available grad for parameter")
                    continue_training = False
                    break

            if not continue_training:
                time.sleep(10)
                model: nn.Module = ps_rref.rpc_sync().get_model().to(self.device)
                continue

            # then sync model
            model: nn.Module = rpc.rpc_sync(
                ps_rref.owner(),
                ParameterServer.update_and_fetch_model,
                args=(ps_rref, rank, [p.grad for p in model.cpu().parameters()]),
            )
            # print("[training] model sync")
            model.to(self.device)

            with lock:
                writer.add_scalar(
                    "training/entropy" + str(rank), entropy_loss.item(), epoch
                )
                writer.add_scalar(
                    "training/value_loss" + str(rank), value_loss.item(), epoch
                )
                writer.add_scalar("training/pg_loss" + str(rank), pg_loss.item(), epoch)
                writer.add_scalar(
                    "training/total_loss" + str(rank), total_loss.item(), epoch
                )
                writer.add_scalar(
                    "training/grad_norm" + str(rank), grad_norm.cpu().item(), epoch
                )
                writer.add_scalar(
                    "training/adv" + str(rank), Advs.detach().mean().item(), epoch
                )
                writer.add_scalar(
                    "training/gae" + str(rank), GAE.detach().mean().item(), epoch
                )
