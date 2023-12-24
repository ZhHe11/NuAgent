from typing import Any, Sequence, Tuple, Dict

import time
import threading

from itertools import count
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from uniagent.trainers.parameter_server import ParameterServer


from torch.distributions import Categorical


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
        model: nn.Module,
        global_counter: mp.Value = None,
        lock: threading.Lock = None,
    ) -> Sequence[Any]:
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

    def test(
        self, args: Namespace, counter: mp.Value, lock: threading.Lock, log_dir: str
    ):
        start_time = time.time()
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in count():
            model: nn.Module = self.ps_rref.rpc_sync().get_model().to(self.device)
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

    def compute_loss(
        self, args: Namespace, model: nn.Module, data: Sequence[Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obses, actions, net_states, rewards, values, log_probs, entropies = data
        episode_len = len(rewards)

        gae = 0.0
        ret = values[-1].item()
        R = [0.0] * episode_len
        GAE = [0.0] * episode_len

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
        actions = torch.from_numpy(np.stack(actions)).squeeze(1).long().to(args.device)
        net_states = [
            torch.from_numpy(np.stack(e)).float().to(args.device) for e in net_states
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
        model: nn.Module = self.ps_rref.rpc_sync().get_model().to(self.device)
        time.sleep(1)
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in count():
            model.train()
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

            with lock:
                writer.add_scalars(
                    "training/episode_info" + str(self.rank),
                    {
                        "episode_reward": sum(rewards),
                        "episode_length": episode_len,
                    },
                    epoch,
                )

            total_loss, loss_detail = self.compute_loss(
                args,
                model,
                (obses, actions, net_states, rewards, values, log_probs, entropies),
            )

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
                self.log_training(epoch, loss_detail, writer)
