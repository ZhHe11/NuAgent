import torch
import torch.nn as nn
import torch.distributed.rpc as rpc

from .parameter_server import ParameterServer


class AsyncTrainer:
    def __init__(self, ps_rref, device):
        self.ps_rref, self.loss_fn = ps_rref, self.create_loss_fn()
        self.device = device

    def create_loss_fn(self):
        raise NotImplementedError

    def get_next_batch(self):
        raise NotImplementedError

    def run(self):
        name = rpc.get_worker_info().name
        # get initial model parameters
        model: nn.Module = self.ps_rref.rpc_sync().get_model().to(self.device)

        while True:
            model = self.step(model)

    def loss_fn(self, model: nn.Module, batch) -> torch.Tensor:
        raise NotImplementedError

    def step(self, model: nn.Module) -> nn.Module:
        # start training
        batch = self.get_next_batch()
        self.loss_fn(model, batch).backward()
        model = rpc.rpc_sync(
            self.ps_rref.owner(),
            ParameterServer.update_and_fetch_model,
            args=(self.ps_rref, [p.grad for p in model.cpu().parameters()]),
        ).to(self.device)

        return model
