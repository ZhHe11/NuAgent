from typing import List, Tuple

import random
import torch

from torch import nn
from torch.nn import functional as F

from uniagent.models.backbone.dilated_lstm import DilatedLSTM
from .utils import reset_grad2


class dLSTM(nn.Module):
    """Implements the dilated LSTM
    Uses a cyclic list of size r to keep r independent hidden states
    """

    def __init__(
        self,
        r: int,
        input_size: int,
        hidden_size: int,
        device: torch.DeviceObjType = torch.device("cpu"),
    ):
        """Construct a dilated LSTM.

        Args:
            r (int): Radius of the dilated LSTM.
            input_size (int): Input dimension.
            hidden_size (int): Hidden size.
            device (torch.DeviceObjType, optional): Device type. Defaults to torch.device("cpu").
        """

        super(dLSTM, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.r = r
        self.device = device
        self.to(self.device)

    def init_state(self, batch_size: int):
        # note that we cannot keep the state in only one tensor as updating one place of the tensor counts
        # as an inplace operation and breaks the gradient history
        hx = [
            torch.zeros(
                batch_size,
                self.lstm.hidden_size,
                requires_grad=self.training,
                device=self.device,
            )
            for _ in range(self.r)
        ]
        cx = [
            torch.zeros(
                batch_size,
                self.lstm.hidden_size,
                requires_grad=self.training,
                device=self.device,
            )
            for _ in range(self.r)
        ]
        tick = 0
        return tick, hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[int, List, List],
        update_network_state: bool = True,
    ):
        """Returns g_t, (tick, hidden)"""

        # tick starts from 0
        tick, hx, cx = states

        # each timestep we update only s^tick_t
        if update_network_state:
            hx[tick], cx[tick] = self.lstm(inputs, (hx[tick], cx[tick]))

            # the output is the average of all hidden states, as being pooled
            out = ((sum(hx) - hx[tick]).detach() + hx[tick]) / self.r

            # update tick here
            tick_plus_one = (tick + 1) % self.r
            states = (tick_plus_one, hx, cx)
        else:
            hx_t, cx_t = self.lstm(inputs, (hx[tick], cx[tick]))
            out = (sum(hx).detach() + hx_t) / self.r
            states = None

        return out, states


class Manager(nn.Module):
    def __init__(
        self, d: int, c: int, device: torch.DeviceObjType = torch.device("cpu")
    ):
        """Construct a manager with a dilated LSTM.

        Args:
            d (int): The dimension of the latent space (s_t in the paper).
            c (int): Time horizon, also the dilation level.
        """

        super(Manager, self).__init__()

        self.c = c
        self.d = d
        self.device = device

        self.f_Mspace = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        # the MRnn can be replaced with a Transformer
        self.f_Mrnn = self.create_goal_embedding()
        self.value_function = self.create_value_function()
        self.to(self.device)

    def create_goal_embedding(self):
        return dLSTM(self.c, self.d, self.d, device=self.device)

    def create_value_function(self):
        return nn.Linear(self.d * 2, 1)

    def forward(
        self,
        z,
        states_M,
        reset_value_grad: bool = False,
        update_network_state: bool = True,
    ):
        s = self.f_Mspace(z)  # latent state representation [batch x d]
        g_hat, states_M = self.f_Mrnn(s, states_M, update_network_state)

        if self.training and random.random() < 0.1:
            # add noise to the g_hat for exploration
            g_hat = g_hat + torch.autograd.Variable(
                g_hat.data.new(g_hat.size()).normal_(0.0, 1.0), requires_grad=False
            )
            # g_hat = torch.autograd.Variable(g_hat.data.new(g_hat.size()).normal_(0.0, 1.0), requires_grad=False)

        g = F.normalize(g_hat)  # goal [batch x d]
        value = self.value_function(
            torch.cat([g, s], dim=-1)
        )  # if reset_value_grad else g_hat)

        return value, g, s.detach(), states_M

    def init_state(self, batch_size: int):
        return self.f_Mrnn.init_state(batch_size)

    def reset_states_grad(self, states):
        tick, hx, cx = states
        return tick, list(map(reset_grad2, hx)), list(map(reset_grad2, cx))


class TransformerManager(Manager):
    def create_goal_embedding(self):
        raise NotImplementedError
