from typing import Tuple, Type
import threading

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torchvision

from torch import optim


num_classes, batch_update_size = 30, 5


def setup_optimizer(args, model) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError
    return optimizer


class ParameterServer:
    def __init__(
        self,
        args,
        model_cls,
        model_kwargs,
        batch_update_size: int = batch_update_size,
        batch_mode: str = "avg",
    ):
        self.args = args
        self.model: nn.Module = model_cls(args=args, **model_kwargs)
        self.model.train()
        self.cnt = 0
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        # NOTE the batch update size would be better for the same as worker number
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.batch_mode = batch_mode

        self.optimizer = self.setup_optimizer()
        self.parameter_buffer = []

        self.reset_grad()

    def setup_optimizer(self):
        optimizer = setup_optimizer(self.args, self.model)
        return optimizer

    def step_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_model(self) -> Tuple[int, nn.Module]:
        return [self.cnt, self.model]

    def reset_grad(self):
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def reset_parameter_buffer(self):
        self.parameter_buffer = [torch.zeros_like(p) for p in self.parameter_buffer]

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, worker_id, grads=None, parameters=None):
        # Using the RRef to retrieve the local PS instance
        self = ps_rref.local_value()

        with self.lock:
            self.curr_update_size += 1
            # accumulate gradients into .grad field
            if grads is not None:
                for p, g in zip(self.model.parameters(), grads):
                    assert g is not None
                    p.grad += g
            elif parameters is not None:
                for i, (_, target_p) in enumerate(
                    zip(self.model.parameters(), parameters)
                ):
                    if len(self.parameter_buffer) == i:
                        self.parameter_buffer.append(torch.zeros_like(target_p))
                    self.parameter_buffer[i] += target_p.data.clone()

            # Save the current future_model and return it to make sure the
            # returned Future object holds the correct model even if another
            # thread modifies future_model before this thread returns.
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                # update the model
                if self.batch_mode == "avg":
                    for p in self.model.parameters():
                        p.grad /= self.batch_update_size
                if grads is not None:
                    self.step_optimizer()
                    self.reset_grad()
                elif parameters is not None:
                    for local_p, target_p in zip(
                        self.model.parameters(), self.parameter_buffer
                    ):
                        local_p.data.copy_(target_p.data / self.batch_update_size)
                    self.reset_parameter_buffer()
                self.curr_update_size = 0
                # by settiing the result on the Future object, all previous
                # requests expecting this updated model will be notified and
                # the their responses will be sent accordingly.
                self.cnt += 1
                fut.set_result(self.model)
                self.future_model = torch.futures.Future()

        return fut


param_server = None
global_lock = threading.Lock()


def get_parameter_server(
    args,
    model_class,
    model_kwargs,
    worker_num,
    batch_mode: str = "sum",
    parameter_server_cls: Type[ParameterServer] = None,
):
    global param_server
    with global_lock:
        if not param_server:
            if parameter_server_cls is None:
                parameter_server_cls = ParameterServer
            param_server = parameter_server_cls(
                args,
                model_class,
                model_kwargs,
                batch_update_size=worker_num,
                batch_mode=batch_mode,
            )
        return param_server


def run_parameter_server(rank: int, world_size: int, ps_name: int):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name=ps_name, rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")
