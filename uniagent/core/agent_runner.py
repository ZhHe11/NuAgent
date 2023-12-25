from typing import Any, Tuple, Dict

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


EpisodeState = namedtuple(
    "EpisodeState",
    "obses, dones, actions, net_states, rewards, values, log_probs, entropies, episode_len",
)


class AgentRunner:
    def __init__(
        self,
        args: Namespace,
        model_class: nn.Module,
        model_kwargs: dict,
        make_env: Any,
        log_dir: str,
    ) -> None:
        self.model = model_class(**model_kwargs).to(args.device)
        self.device = args.device
        self.env = make_env()
        self.writer = SummaryWriter(log_dir=log_dir)

    def run_episode(
        self,
        args: Namespace,
        obs: np.ndarray,
        model: nn.Module,
        global_counter: mp.Value = None,
        lock: threading.Lock = None,
    ) -> EpisodeState:
        done = False

        rewards = []
        values = []
        log_probs = []
        entropies = []
        obses = []
        actions = []
        dones = []
        net_states = []

        # make sure it is not None
        net_state = model.init_state(1, self.device)

        counter = count() if model.training else range(args.num_steps)

        for step_cnt in counter:
            obses.append(obs)
            net_states.append([e.squeeze(0) for e in net_state])
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
            obs = torch.from_numpy(obs).float()
            value, _, _, _, _ = self.act(obs.unsqueeze(0), net_state)
            values.append(value.squeeze())

        return EpisodeState(
            obses,
            dones,
            actions,
            net_states,
            rewards,
            values,
            log_probs,
            entropies,
            len(rewards),
        )

    def update_and_fetch_model(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    def fetch_model(self) -> nn.Module:
        return self.model

    def test(self, args: Namespace, counter: mp.Value, lock: threading.Lock):
        start_time = time.time()

        for epoch in count():
            model: nn.Module = self.ps_rref.rpc_sync().get_model().to(self.device)
            model.eval()
            # always reset
            obs, _ = self.env.reset()
            # test should not update counter
            episode_state = self.run_episode(args, obs, model)
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
                for name, param in model.named_parameters():
                    self.writer.add_histogram(
                        name, param.clone().cpu().data.numpy(), epoch
                    )
                self.writer.add_scalar(
                    "evaluation/episode_reward", reward_sum, counter.value
                )
            time.sleep(5)

    def compute_loss(
        self, args: Namespace, model: nn.Module, episode_state: EpisodeState
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def log_training(
        self, epoch: int, loss_detail: Dict[str, torch.Tensor], writer: SummaryWriter
    ):
        pass

    def train(
        self, args: Namespace, counter: mp.Value = None, lock: threading.Lock = None
    ):
        self.model: nn.Module = self.fetch_model()

        obs, _ = self.env.reset()

        for epoch in count():
            model.train()
            episode_state = self.run_episode(args, obs, model, counter, lock)

            with lock:
                self.writer.add_scalars(
                    "training/episode_info" + str(self.rank),
                    {
                        "episode_reward": sum(episode_state.rewards),
                        "episode_length": episode_state.episode_len,
                    },
                    epoch,
                )

            total_loss, loss_detail = self.compute_loss(args, model, episode_state)

            assert total_loss.requires_grad

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )

            loss_detail["grad_norm"] = grad_norm

            # then sync model
            model = self.update_and_fetch_model(model)
            assert model.training

            with lock:
                self.log_training(epoch, loss_detail, self.writer)

            if episode_state.dones[-1]:
                obs, _ = self.env.reset()
            else:
                obs = episode_state.obses[-1]
