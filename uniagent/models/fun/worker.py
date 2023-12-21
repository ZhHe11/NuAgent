import torch

from torch import nn
from torch.nn import functional as F

from .utils import reset_grad2, View


class Worker(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        d: int,
        k: int,
        device: torch.DeviceObjType = torch.device("cpu"),
    ):
        """Construct a Worker

        Args:
            num_outputs (int): Output number
            d (int): ...
            k (int): The dimension size of the embedded goals
        """

        super(Worker, self).__init__()

        self.num_outputs = num_outputs
        self.k = k
        self.d = d
        self.device = device

        self.f_Wrnn = self.create_observation_embedding()

        self.view_as_actions = View((k, num_outputs))

        self.phi = nn.Sequential(nn.Linear(d, k, bias=False), View((1, k)))

        self.value_function = self.create_value_function()

        self.to(self.device)

    def create_value_function(self):
        return nn.Sequential(
            nn.Linear(self.num_outputs * self.k, 50), nn.ReLU(), nn.Linear(50, 1)
        )

    def create_observation_embedding(self):
        lstm = nn.LSTMCell(self.d, self.num_outputs * self.k)
        lstm.bias_ih.data.fill_(0)
        lstm.bias_hh.data.fill_(0)
        return lstm

    def reset_states_grad(self, states):
        h, c = states
        return reset_grad2(h), reset_grad2(c)

    def init_state(self, batch_size):
        return (
            torch.zeros(
                batch_size,
                self.f_Wrnn.hidden_size,
                requires_grad=self.training,
                device=self.device,
            ),
            torch.zeros(
                batch_size,
                self.f_Wrnn.hidden_size,
                requires_grad=self.training,
                device=self.device,
            ),
        )

    def forward(self, z, sum_g_W, states_W, reset_value_grad: bool = True):
        """
        :param z:
        :param sum_g_W: should not have computation history
        :param worker_states:
        :return:
        """

        # project the last c goals into a single vector, where the dimension
        #   size is k
        w = self.phi(sum_g_W)  # projection [ batch x 1 x k]

        # Worker firstly embeds the input observations
        _, cx = states_W = self.f_Wrnn(z, states_W)
        U = self.view_as_actions(cx)  # [batch x k x a]
        # then coordinates the embedded observation with the embedded goals
        a = (w @ U).squeeze(1)  # [batch x a]

        probs = F.softmax(a, dim=1)
        value = self.value_function(cx.detach() if reset_value_grad else cx)

        return value, probs, states_W


class TransformerWorker(Worker):
    def create_observation_embedding(self):
        raise NotImplementedError
