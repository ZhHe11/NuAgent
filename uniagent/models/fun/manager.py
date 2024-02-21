from typing import List, Tuple

import random
import torch

from torch import DeviceObjType, nn
from torch._C import DeviceObjType, device
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
    def __init__(self, observation_space, action_space, config):
        """Construct a manager with a dilated LSTM.

        Args:
            d (int): The dimension of the latent space (s_t in the paper).
            c (int): Time horizon, also the dilation level.
        """

        super(Manager, self).__init__()

        self.device = config.device
        self.config = config

        self.f_Mspace = (
            self.create_state_embedding()
        )  # nn.Sequential(nn.Linear(self.config.d, self.config.d), nn.ReLU())
        # the MRnn can be replaced with a Transformer
        self.f_Mrnn = self.create_goal_embedding()
        self.value_function = self.create_value_function()
        self.to(self.device)

    def create_state_embedding(self):
        return nn.Sequential(nn.Linear(self.config.d, self.config.d), nn.ReLU())

    def create_goal_embedding(self):
        return dLSTM(self.config.c, self.config.d, self.config.d, device=self.device)

    def create_value_function(self):
        return nn.Linear(self.config.d * 2, 1)

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


from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from uniagent.models.mingpt import blocks
from argparse import Namespace


class TransformerManager(Manager):
    def __init__(self, observation_space, action_space, config: Namespace):
        super().__init__(observation_space, action_space, config)
        self.transformer = self.f_Mrnn
        self.tokenizer = self.create_tokenizer()
        self.action_decoder = self.create_action_decoder()

    def create_state_embedding(self):
        pass

    def create_value_function(self):
        return nn.Linear(self.config.n_embed, 1, bias=False)

    def create_tokenizer(self):
        pass

    def create_action_decoder(self):
        pass

    def create_goal_embedding(self):
        transformer = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(self.config.seq_length, self.config.n_embed),
                drop=nn.Dropout(self.config.embed_pdrop),
                h=nn.ModuleList(
                    [blocks.Block(self.config) for _ in range(self.config.n_layer)]
                ),
                ln_f=nn.LayerNorm(self.config.n_embed),
                lm_head_loc=nn.Linear(
                    self.config.n_embed, self.config.vocab_size, bias=False
                ),
                lm_head_scale=nn.Linear(
                    self.config.n_embed, self.config.vocab_size, bias=False
                ),
            )
        )
        return transformer

    def init_state(self, batch_size: int):
        return (
            torch.zeros(
                batch_size,
                requires_grad=False,
                device=self.device,
            ),
            torch.zeros(
                batch_size,
                requires_grad=False,
                device=self.device,
            ),
        )

    def forward(self, token_seq_embedding: torch.Tensor, state: torch.Tensor):
        # g_hat: [batch_size, seq_len, vocab_size]
        batch_size, t, _ = token_seq_embedding.size()
        # [1, seq_len]
        pos = torch.arange(0, t, dtype=torch.long, device=self.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(token_seq_embedding + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        g_loc = self.transformer.lm_head_loc(x)
        g_scale = torch.exp(self.transformer.lm_head_scale(x))
        g_cov = torch.diag_embed(g_scale)
        dist = MultivariateNormal(g_loc, g_cov)

        if self.training:
            g_hat = dist.sample()
        else:
            g_hat = g_loc

        g_log_prob = dist.log_prob(g_hat)
        g_entropy = dist.entropy()
        value = self.value_function(x)

        return value, g_hat, g_log_prob, g_entropy, state
