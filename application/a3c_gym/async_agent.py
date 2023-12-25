from typing import Any, Tuple, Dict

from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc

from tensorboardX import SummaryWriter

from uniagent.trainers.parameter_server import ParameterServer
from uniagent.core.agent_runner import AgentRunner, EpisodeState


class AsyncAgent(AgentRunner):
    def __init__(
        self,
        args: Namespace,
        ps_rref: Any,
        rank: int,
        model_class: nn.Module,
        model_kwargs: dict,
        make_env: Any,
        log_dir: str,
    ) -> None:
        self.ps_rref = ps_rref
        self.rank = rank
        self.worker_name = rpc.get_worker_info().name
        super().__init__(args, model_class, model_kwargs, make_env, log_dir)

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

    def compute_loss(
        self, episode_state: EpisodeState
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        gae = 0.0
        ret = episode_state.values[-1].item()
        R = [0.0] * episode_state.episode_len
        GAE = [0.0] * episode_state.episode_len

        for i in reversed(range(episode_state.episode_len)):
            ret = self.args.gamma * ret + episode_state.rewards[i]
            delta_t = (
                episode_state.rewards[i]
                + self.args.gamma * episode_state.values[i + 1].cpu().item()
                - episode_state.values[i].cpu().item()
            )
            gae = gae * self.args.gamma * self.args.llambda + delta_t

            GAE[i] = gae
            R[i] = ret

        R = torch.from_numpy(np.asarray(R)).float().to(self.device)
        GAE = torch.from_numpy(np.asarray(GAE)).float().to(self.device)
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
            + self.args.value_loss_coef * value_loss
            - self.args.entropy_coef * entropy_loss
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
