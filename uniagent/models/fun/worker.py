from typing import Tuple
from argparse import Namespace

import torch

from torch import nn
from torch.nn import functional as F

from .utils import reset_grad2


class Worker(nn.Module):
    def __init__(self, config):
        """Construct a Worker

        Args:
            num_outputs (int): Output number
            d (int): ...
            k (int): The dimension size of the embedded goals
        """

        super(Worker, self).__init__()

        self.num_outputs = config.num_outputs
        self.k = config.k
        self.d = config.d
        self.device = config.device
        self.config = config

        self.f_Wrnn = self.create_observation_embedding()
        self.proj = nn.Linear(config.k, config.k, bias=False)
        self.value_function = self.create_value_function()

        self.to(self.device)

    def create_value_function(self):
        return nn.Sequential(
            nn.Linear(self.num_outputs * self.k + self.k, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
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

    def forward(
        self,
        z: torch.Tensor,
        sum_g_W: torch.Tensor,
        states_W: Tuple[torch.Tensor, torch.Tensor],
        reset_value_grad: bool = True,
        update_network_state: bool = True,
    ):
        """
        :param z:
        :param sum_g_W: should not have computation history
        :param worker_states:
        :return:
        """

        # project the last c goals into a single vector, where the dimension
        #   size is k
        w = self.proj(sum_g_W)  # projection [ batch x k]

        # Worker firstly embeds the input observations
        states_W = self.f_Wrnn(z, states_W)
        hx, cx = states_W
        U = hx.reshape(hx.shape[0], self.k, self.num_outputs)
        # then coordinates the embedded observation with the embedded goals
        a = torch.einsum("bk,bka->ba", w, U)  # [batch x a]
        probs = F.softmax(a, dim=1)
        value = self.value_function(
            torch.cat([cx, w], dim=-1)
        )  # if reset_value_grad else hx)

        if not update_network_state:
            states_W = None

        return value, probs, states_W


from uniagent.models.mingpt import OuterQueryGPT


class TransformerWorker(Worker):
    def __init__(self, observation_space, action_space, config: Namespace):
        self.observation_space = observation_space
        self.action_space = action_space
        super().__init__(config)
        self.outer_query_gpt = self.f_Wrnn

    def create_value_function(self):
        return nn.Linear(self.config.n_embed, 1, bias=False)

    def create_observation_embedding(self):
        raise OuterQueryGPT(self.observation_space, self.action_space, self.config)

    def forward(
        self,
        token_seq_emb: torch.Tensor,
        queries: torch.Tensor,
        states_W: torch.Tensor,
    ):
        # logits: [batch_size, seq_len, vocab_size]
        logits, states_W = self.outer_query_gpt(
            token_seq_emb, queries=queries, states=states_W
        )
        values = self.value_function(states_W)
        return values, logits, states_W
