from typing import Any, Tuple, Dict

from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc

from tensorboardX import SummaryWriter

from uniagent.trainers.parameter_server import ParameterServer
from uniagent.core.agent_runner import AgentRunner, EpisodeState


def compute_gae_and_ret(
    args: Namespace,
    model: nn.Module,
    episode_state: EpisodeState,
    recompute_value: bool = False,
) -> EpisodeState:
    new_episode_state = EpisodeState(episode_state.__dict__)
    R = [0.0] * episode_state.episode_len
    GAE = [0.0] * episode_state.episode_len
    # gae = 0.0

    if recompute_value:
        values, logits, _ = model(new_episode_state.obses, new_episode_state.net_states)
        next_values, _, _ = model(
            new_episode_state.next_obses, new_episode_state.next_net_states
        )
        new_episode_state.logits = logits
        new_episode_state.state_values = values
        new_episode_state.next_state_values = next_values

    # delta_ts = (
    #     (
    #         new_episode_state.rewards
    #         + args.gamma * new_episode_state.next_state_values
    #         - new_episode_state.state_values
    #     )
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )

    # R = (
    #     np.frompyfunc(lambda x, y: args.gamma * x + y, 2, 1)
    #     .accumulate(np.flip(episode_state.rewards.cpu().numpy(), axis=0))
    #     .tolist()
    # )
    # GAE = (
    #     np.frompyfunc(lambda x, y: args.gamma * args.llambda * x + y, 2, 1)
    #     .accumulate(np.flip(delta_ts, axis=0))
    #     .tolist()
    # )
    next_state_values = new_episode_state.next_state_values.detach().cpu().numpy()
    state_values = new_episode_state.state_values.detach().cpu().numpy()
    rewards = new_episode_state.rewards.cpu().numpy()
    ret = next_state_values[-1]
    gae = 0.0
    for i in reversed(range(new_episode_state.episode_len)):
        ret = args.gamma * ret + rewards[i]
        delta_t = rewards[i] + args.gamma * next_state_values[i] - state_values[i]
        gae = gae * args.gamma * args.llambda + delta_t

        GAE[i] = gae
        R[i] = ret

    R = torch.FloatTensor(R).to(args.device)
    GAE = torch.FloatTensor(GAE).to(args.device)
    ADV = R - new_episode_state.state_values

    new_episode_state.rets = R
    new_episode_state.gae = GAE
    new_episode_state.adv = ADV

    return new_episode_state


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
        model_cnt, model = self.ps_rref.rpc_sync().get_model()
        return model.to(self.device)

    def handle_net_states(self, episode_state: EpisodeState) -> Any:
        return tuple([e[-1] for e in episode_state.net_states])

    def preprocess_episode(self, episode_state: EpisodeState):
        new_episode_state = EpisodeState(episode_state.__dict__)
        obses = (
            torch.from_numpy(np.stack(new_episode_state.obses))
            .float()
            .squeeze(1)
            .to(self.device)
        )
        new_episode_state.obses = obses[:-1]
        new_episode_state.next_obses = obses[1:]
        net_states = tuple(
            map(lambda x: torch.stack(x).squeeze(1), zip(*new_episode_state.net_states))
        )
        new_episode_state.net_states = tuple([e[:-1] for e in net_states])
        new_episode_state.next_net_states = tuple([e[1:] for e in net_states])

        values = torch.stack(new_episode_state.state_values)
        assert len(values.shape) == 1
        new_episode_state.state_values = values[:-1]
        new_episode_state.next_state_values = values[1:]
        new_episode_state.entropies = torch.stack(episode_state.entropies)
        new_episode_state.log_probs = torch.stack(new_episode_state.log_probs)
        new_episode_state.rewards = (
            torch.from_numpy(np.stack(new_episode_state.rewards))
            .float()
            .to(self.device)
        )

        return new_episode_state

    def compute_loss(
        self, episode_state: EpisodeState
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        episode_state = self.preprocess_episode(episode_state)
        episode_state = compute_gae_and_ret(self.args, self.model, episode_state)
        R = episode_state.rets
        Advs = episode_state.adv
        GAE = episode_state.gae

        log_probs = episode_state.log_probs
        entropies = episode_state.entropies
        values = episode_state.state_values

        assert values.shape == R.shape, (
            values.shape,
            R.shape,
        )

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
