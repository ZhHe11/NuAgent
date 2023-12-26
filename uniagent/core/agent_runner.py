from typing import Any, Tuple, Dict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import time
import threading

from itertools import count
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter


from torch.distributions import Categorical


# EpisodeState = namedtuple(
#     "EpisodeState",
#     "obses, dones, actions, net_states, rewards, values, log_probs, entropies, episode_len",
# )


class EpisodeState(dict):
    def __init__(self, episode_state: dict = None, copy: bool = False, **kwargs):
        super().__init__()
        if copy:
            episode_state = deepcopy(episode_state)
        if episode_state is not None:
            assert isinstance(episode_state, dict)
            for k, v in episode_state.items():
                self.__dict__[k] = v
        if len(kwargs):
            self.__init__(kwargs, copy=copy)

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value

    def __getattr__(self, __key: str):
        return self.__dict__[__key]

    def __setitem__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value

    def __getitem__(self, __key: Any) -> Any:
        return self.__dict__[__key]


class AgentRunner:
    def __init__(
        self,
        args: Namespace,
        model_class: nn.Module,
        model_kwargs: dict,
        make_env: Any,
        log_dir: str,
    ) -> None:
        self.args = args
        self.model = model_class(**model_kwargs).to(args.device)
        self.device = args.device
        self.env = make_env()
        self.log_dir = log_dir

    def run_episode(
        self,
        obs: np.ndarray,
        done: bool,
        net_state: Any,
        global_counter: mp.Value = None,
        lock: threading.Lock = None,
    ) -> EpisodeState:
        rewards = []
        values = []
        log_probs = []
        entropies = []
        obses = []
        actions = []
        dones = []
        net_states = []

        # make sure it is not None
        if done:
            net_state = self.model.init_state(1, self.device)
        else:
            net_state = self.model.reset_states_grad(net_state)

        counter = count() if not self.model.training else range(self.args.num_steps)

        for step_cnt in counter:
            obses.append(obs)
            net_states.append([e.squeeze(0) for e in net_state])
            obs = torch.from_numpy(obs).float().to(self.device)
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
            dones.append(done)

            if global_counter:
                with lock:
                    global_counter.value += 1

            if done:
                break

        obses.append(obs)
        net_states.append([e.squeeze(0) for e in net_state])

        if dones[-1]:
            values.append(torch.zeros(1).to(self.device).squeeze())
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
            value, _, _ = self.model(obs.unsqueeze(0), net_state)
            values.append(value.squeeze())

        return EpisodeState(
            obses=obses,
            dones=dones,
            actions=actions,
            net_states=net_states,
            rewards=rewards,
            state_values=values,
            log_probs=log_probs,
            entropies=entropies,
            episode_len=len(rewards),
        )

    def update_and_fetch_model(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    def fetch_model(self) -> nn.Module:
        raise NotImplementedError

    def test(self, counter: mp.Value, lock: threading.Lock):
        start_time = time.time()
        writer = SummaryWriter(log_dir=self.log_dir)

        for epoch in count():
            self.model: nn.Module = self.fetch_model()
            self.model.eval()
            # always reset
            obs, _ = self.env.reset()
            # test should not update counter
            episode_state = self.run_episode(obs, done=True, net_state=None)
            reward_sum = sum(episode_state.rewards)
            print(
                "Time {}, eval epoch {}, training steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    epoch,
                    counter.value,
                    counter.value / (time.time() - start_time),
                    reward_sum,
                    episode_state.episode_len,
                )
            )
            with lock:
                for name, param in self.model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                writer.add_scalar(
                    "evaluation/episode_reward", reward_sum, counter.value
                )
            time.sleep(5)

    def compute_loss(
        self, episode_state: EpisodeState
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def log_training(
        self, epoch: int, loss_detail: Dict[str, torch.Tensor], writer: SummaryWriter
    ):
        pass

    def train(self, counter: mp.Value = None, lock: threading.Lock = None):
        self.model: nn.Module = self.fetch_model()
        time.sleep(1)
        writer = SummaryWriter(log_dir=self.log_dir)

        obs, _ = self.env.reset()
        last_done = True
        last_net_states = None

        for epoch in count():
            self.model.train()
            episode_state = self.run_episode(
                obs, last_done, last_net_states, counter, lock
            )

            with lock:
                writer.add_scalars(
                    "training/episode_info" + str(self.rank),
                    {
                        "episode_reward": sum(episode_state.rewards),
                        "episode_length": episode_state.episode_len,
                    },
                    epoch,
                )

            total_loss, loss_detail = self.compute_loss(episode_state)

            assert total_loss.requires_grad

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm
            )

            loss_detail["grad_norm"] = grad_norm

            # then sync model
            self.model = self.update_and_fetch_model(self.model)
            assert self.model.training

            with lock:
                self.log_training(epoch, loss_detail, writer)

            if episode_state.dones[-1]:
                obs, _ = self.env.reset()
            else:
                obs = episode_state.obses[-1]
            last_done = episode_state.dones[-1]
            last_net_states = self.handle_net_states(episode_state)

    def handle_net_states(self, episode_state: EpisodeState) -> Any:
        return episode_state.net_states
