from typing import Tuple
from argparse import Namespace

import torch
import torch.nn as nn

from .gpt import GPT
from .blocks import Block, CausalSelfAttention


class CausalExternalAttention(CausalSelfAttention):
    pass


class OuterQueryBlock(Block):
    def create_attn(self, config):
        return CausalExternalAttention(config)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), self.ln_1(query))
        x = x + self.mlpf(self.ln_2(x))
        return x


class OuterQueryGPT(GPT):
    def __init__(self, observation_space, action_space, config: Namespace):
        self.observation_space = observation_space
        self.action_space = action_space
        super().__init__(config)

    def construct_transformer(self, config: Namespace):
        transformer = nn.ModuleDict(
            dict(
                wpt=nn.Embedding(config.seq_length, config.n_embed),
                drop=nn.Dropout(config.embed_pdrop),
                h=nn.ModuleList(
                    [OuterQueryBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        return transformer

    def construct_lm_head(self, config: Namespace) -> nn.Module:
        return nn.Linear(config.n_embed, self.action_space.n, bias=False)

    def forward(
        self,
        tok_seq_emb: torch.Tensor,
        queries: torch.Tensor,
        states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = tok_seq_emb.device
        b, t, _ = tok_seq_emb.shape.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)
        pos_emb = self.transformer.wpt(pos)
        x = self.transformer.drop(tok_seq_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, queries)
        # [batch_size, seq_len, n_embed]
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, x
