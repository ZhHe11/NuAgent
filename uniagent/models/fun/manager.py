import torch

from torch import nn
from torch.nn import functional as F

from .utils import reset_grad2


class dLSTM(nn.Module):
    """Implements the dilated LSTM
    Uses a cyclic list of size r to keep r independent hidden states
    """

    def __init__(self, r, input_size, hidden_size):
        super(dLSTM, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.r = r

    def init_state(self, batch_size):
        # note that we cannot keep the state in only one tensor as updating one place of the tensor counts
        # as an inplace operation and breaks the gradient history
        h0 = [
            torch.zeros(batch_size, self.lstm.hidden_size, requires_grad=self.training)
            for _ in range(self.r)
        ]
        c0 = [
            torch.zeros(batch_size, self.lstm.hidden_size, requires_grad=self.training)
            for _ in range(self.r)
        ]
        tick = 0
        return tick, h0, c0

    def forward(self, inputs, states):
        """Returns g_t, (tick, hidden)"""
        tick, hx, cx = states
        hx[tick], cx[tick] = self.lstm(inputs, (hx[tick], cx[tick]))
        tick = (tick + 1) % self.r
        out = (
            sum(hx) / self.r
        )  # TODO verify that network output is mean of hidden states
        return out, (tick, hx, cx)


class Manager(nn.Module):
    def __init__(self, d, c):
        super(Manager, self).__init__()
        self.c = c

        self.f_Mspace = nn.Sequential(nn.Linear(d, d), nn.ReLU())

        self.f_Mrnn = dLSTM(c, d, d)

        self.value_function = nn.Linear(d, 1)

    def forward(self, z, states_M, reset_value_grad):
        s = self.f_Mspace(z)  # latent state representation [batch x d]
        g_hat, states_M = self.f_Mrnn(s, states_M)

        g = F.normalize(g_hat)  # goal [batch x d]

        if reset_value_grad:
            value = self.value_function(reset_grad2(g_hat))
        else:
            value = self.value_function(g_hat)

        return value, g, s, states_M

    def init_state(self, batch_size):
        return self.f_Mrnn.init_state(batch_size)

    def reset_states_grad(self, states):
        tick, hx, cx = states
        return tick, list(map(reset_grad2, hx)), list(map(reset_grad2, cx))
