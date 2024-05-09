from typing import Type, Dict, Any, Tuple, Sequence

from copy import deepcopy

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from torchvision.ops import MLP


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
        self.token_dim = token_dim

        self.experts = self.create_experts()

        # [D, N * P]
        weights = np.random.random((token_dim, num_experts * num_slots))
        self.register_parameter(
            "expert_mixture",
            nn.Parameter(torch.tensor(weights).float(), requires_grad=True),
        )

    def create_experts(self) -> nn.ModuleList:
        experts = []
        if self.base_model_cls is None:
            for _ in range(self.n):
                expert = MLP(
                    self.token_dim,
                    hidden_channels=[256, 256, self.token_dim],
                    activation_layer=nn.ReLU,
                )
                experts.append(expert)
        return nn.ModuleList(experts)

    def cal_token_mat(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B * T, D]
        tokens = tokens.reshape(-1, self.expert_mixture.size(0))
        # [B * T, N * P] = [M, N * P]
        mat = tokens.matmul(self.expert_mixture)
        return mat, tokens

    def cal_raw_probs(self, mat: torch.Tensor) -> torch.Tensor:
        # mat shape: [M, N * P]
        probs = F.softmax(mat, dim=0)
        return probs

    def cal_col_probs(self, mat: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(mat, dim=1)
        return probs

    def apply_experts(self, xs, ys) -> torch.Tensor:
        for i in range(self.n):
            ys[i] = self.experts[i](xs[i])
        y = torch.row_stack(ys)
        return y

    def forward(self, tokens: torch.Tensor):
        # B: batch_size, T: num_token, D: token_dim
        # token: [B, T, D]
        if tokens.ndim == 2:
            batch_of_token_seq = tokens.unsqueeze(0)  # [1, T, D]
        else:
            batch_of_token_seq = tokens

        # mixture of token at row-axis for each slots
        B, T, D = batch_of_token_seq.size()
        mat, reshaped_tokens = self.cal_token_mat(batch_of_token_seq)
        # [B * T, N * P]
        row_probs = self.cal_raw_probs(mat)
        # [B * T, N * P]
        col_probs = self.cal_col_probs(mat)
        # [N * P, D]
        x_hat = row_probs.T.matmul(reshaped_tokens)
        # apply experts for each slots
        xs = x_hat.split(self.p)
        ys = [0] * self.n
        y_hat = self.apply_experts(xs, ys)
        # [B * T, D]
        y = col_probs.matmul(y_hat)
        y = y.view_as(tokens)
        return y
