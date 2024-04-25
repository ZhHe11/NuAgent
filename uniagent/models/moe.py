from typing import Type, Dict, Any, Tuple, Sequence

from copy import deepcopy

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F


class SoftMoELayer(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_experts: int,
        num_slots: int = 1,
        base_model_cls: Type = None,
        base_model_kwargs: Dict[str, Any] = None,
    ):
        super().__init__()
        self.base_model_cls = base_model_cls
        self.base_model_kwargs = deepcopy(base_model_kwargs)
        self.n = num_experts
        self.p = num_slots

        # [D, N * P]
        weights = np.random.random(token_dim, num_experts * num_slots)
        self.weights = nn.Parameter(torch.tensor(weights), requires_grad=True)
        self.experts = self.create_experts()

    def create_experts(self) -> nn.ModuleList:
        raise NotImplementedError

    def cal_token_mat(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B * T, D]
        tokens = tokens.reshape(-1, self.weights.size(0))
        # [B * T, N * P] = [M, N * P]
        mat = tokens.matmul(self.weights)
        return mat, tokens

    def cal_raw_probs(self, mat: torch.Tensor) -> torch.Tensor:
        # mat shape: [M, N * P]
        probs = F.softmax(mat, dim=0)
        return probs

    def cal_col_probs(self, mat: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(mat, dim=1)
        return probs

    @torch.jit
    def apply_experts(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        ys = [0] * self.n
        for i in range(self.n):
            ys[i] = self.experts[i](xs[i])
        y = torch.row_stack(ys)
        return y

    def forward(self, tokens: torch.Tensor):
        # B: batch_size, T: num_token, D: token_dim
        # token: [B, T, D]
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)  # [1, T, D]

        # mixture of token at row-axis for each slots
        B, T, D = tokens.size()
        mat, reshaped_tokens = self.cal_token_mat(tokens)
        # [B * T, N * P]
        row_probs = self.cal_raw_probs(mat)
        # [B * T, N * P]
        col_probs = self.cal_col_probs(mat)
        # [N * P, D]
        x_hat = row_probs.T.matmul(reshaped_tokens)
        # apply experts for each slots
        xs = x_hat.split(self.p)

        y_hat = self.apply_experts(xs)
        # [B * T, D]
        y = col_probs.matmul(y_hat)
        y = y.reshape(B, T, D)
        return y
