from typing import Any, Sequence, Tuple, Dict

import time
import threading

from itertools import count
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from uniagent.trainers.parameter_server import ParameterServer


from torch.distributions import Categorical


EpisodeState = namedtuple(
    "EpisodeState",
    "obses, dones, actions, net_states, rewards, values, log_probs, entropies, episode_len",
)


class AsyncAgent:
    def __init__(
        self,
        ps_rref: Any,
        rank: int,
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
        self.ps_rref = ps_rref
        self.rank = rank

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
            value, _, _ = self.model(obs.unsqueeze(0), net_state)
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
        for p in model.parameters():
            if p.grad is None:
                raise RuntimeError(
                    f"Empty grad at worker: {self.worker_name} {self.rank}"
                )

        model: nn.Module = rpc.rpc_sync(
            self.ps_rref.owner(),
            ParameterServer.update_and_fetch_model,
            args=(
                self.ps_rref,
                self.worker_name,
                [p.grad for p in model.cpu().parameters()],
            ),
        ).to(self.device)
        return model

    def fetch_model(self) -> nn.Module:
        return self.ps_rref.rpc_sync().get_model().to(self.device)

    def test(
        self, args: Namespace, counter: mp.Value, lock: threading.Lock, log_dir: str
    ):
        start_time = time.time()
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in count():
            self.model: nn.Module = self.fetch_model()
            self.model.eval()
            # always reset
            obs, _ = self.env.reset()
            episode_state = self.run_episode(args, obs, self.model)
            reward_sum = sum(episode_state.rewards)
            print(
                "Time {}, epoch {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
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
        self, args: Namespace, model: nn.Module, episode_state: EpisodeState
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # obses, dones, actions, net_states, rewards, values, log_probs, entropies = episode_state
        gae = 0.0
        ret = episode_state.values[-1].item()
        R = [0.0] * episode_state.episode_len
        GAE = [0.0] * episode_state.episode_len

        for i in reversed(range(episode_state.episode_len)):
            ret = args.gamma * ret + episode_state.rewards[i]
            delta_t = (
                episode_state.rewards[i]
                + args.gamma * episode_state.values[i + 1].cpu().item()
                - episode_state.values[i].cpu().item()
            )
            gae = gae * args.gamma * args.llambda + delta_t

            GAE[i] = gae
            R[i] = ret

        R = torch.from_numpy(np.asarray(R)).float().to(args.device)
        GAE = torch.from_numpy(np.asarray(GAE)).float().to(args.device)
        log_probs = torch.stack(episode_state.log_probs)
        entropies = torch.stack(episode_state.entropies)
        values = torch.stack(episode_state.values[:-1])

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
        loss_detail = {
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "pg_loss": pg_loss,
            "total_loss": total_loss,
            "Advs": Advs,
            "GAE": GAE,
        }
        return total_loss, loss_detail

    def log_training(
        self, epoch: int, loss_detail: Dict[str, torch.Tensor], writer: SummaryWriter
    ):
        writer.add_scalar(
            "training/entropy" + str(self.rank),
            loss_detail["entropy_loss"].item(),
            epoch,
        )
        writer.add_scalar(
            "training/value_loss" + str(self.rank),
            loss_detail["value_loss"].item(),
            epoch,
        )
        writer.add_scalar(
            "training/pg_loss" + str(self.rank), loss_detail["pg_loss"].item(), epoch
        )
        writer.add_scalar(
            "training/total_loss" + str(self.rank),
            loss_detail["total_loss"].item(),
            epoch,
        )
        writer.add_scalar(
            "training/grad_norm" + str(self.rank),
            loss_detail["grad_norm"].cpu().item(),
            epoch,
        )
        writer.add_scalar(
            "training/adv" + str(self.rank),
            loss_detail["Advs"].detach().mean().item(),
            epoch,
        )
        writer.add_scalar(
            "training/gae" + str(self.rank),
            loss_detail["GAE"].detach().mean().item(),
            epoch,
        )

    def train(
        self, args: Namespace, counter: mp.Value, lock: threading.Lock, log_dir: str
    ):
        self.model: nn.Module = self.fetch_model()
        time.sleep(1)
        writer = SummaryWriter(log_dir=log_dir)

        obs, _ = self.env.reset()

        for epoch in count():
            self.model.train()
            episode_state = self.run_episode(args, obs, self.model, counter, lock)

            with lock:
                writer.add_scalars(
                    "training/episode_info" + str(self.rank),
                    {
                        "episode_reward": sum(episode_state.rewards),
                        "episode_length": episode_state.episode_len,
                    },
                    epoch,
                )

            total_loss, loss_detail = self.compute_loss(args, self.model, episode_state)

            assert total_loss.requires_grad

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), args.max_grad_norm
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
