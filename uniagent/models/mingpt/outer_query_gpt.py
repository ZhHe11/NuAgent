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
    def construct_transformer(self, config: Namespace):
        transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList(
                    [OuterQueryBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        return transformer

    def compute_with_outer_embedding(
        self, tok_emb: torch.Tensor, queries: torch.Tensor, states: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = tok_emb.device
        b, t, _ = tok_emb.shape.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, queries)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        values = self.value_function(x)
        return values, logits, states

    def forward(
        self, x: torch.Tensor, query: torch.Tensor, states: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # note: the input x should be of shape [batch x block_size]
        # note: the input query should be of shape [batch x query_size]
        # note: the output is of shape [batch x block_size x n_embd]
        raise NotImplementedError
