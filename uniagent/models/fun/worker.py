import torch

from torch import nn
from torch.nn import functional as F

from .utils import reset_grad2, View


class Worker(nn.Module):
    def __init__(self, num_outputs, d, k):
        super(Worker, self).__init__()

        self.f_Wrnn = nn.LSTMCell(d, num_outputs * k)

        self.view_as_actions = View((k, num_outputs))

        self.phi = nn.Sequential(nn.Linear(d, k, bias=False), View((1, k)))

        self.value_function = nn.Linear(num_outputs * k, 1)

    def reset_states_grad(self, states):
        h, c = states
        return reset_grad2(h), reset_grad2(c)

    def init_state(self, batch_size):
        return (
            torch.zeros(
                batch_size, self.f_Wrnn.hidden_size, requires_grad=self.training
            ),
            torch.zeros(
                batch_size, self.f_Wrnn.hidden_size, requires_grad=self.training
            ),
        )

    def forward(self, z, sum_g_W, states_W, reset_value_grad):
        """
        :param z:
        :param sum_g_W: should not have computation history
        :param worker_states:
        :return:
        """
        w = self.phi(sum_g_W)  # projection [ batch x 1 x k]

        # Worker
        U_flat, c_x = states_W = self.f_Wrnn(z, states_W)
        U = self.view_as_actions(U_flat)  # [batch x k x a]

        a = (w @ U).squeeze(1)  # [batch x a]

        probs = F.softmax(a, dim=1)

        if reset_value_grad:
            value = self.value_function(reset_grad2(U_flat))
        else:
            value = self.value_function(U_flat)

        return value, probs, states_W
