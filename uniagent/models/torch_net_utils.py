import math
import torch

from torch import nn


class View(nn.Module):
    """Layer changing the tensor shape
    Assumes batch first tensors by default
    Args:
        the output shape without providing the batch size
    """

    def __init__(self, shape, batched=True):
        super(View, self).__init__()
        self.shape = shape
        self.batched = batched

    def forward(self, x):
        if self.batched:
            return x.view(-1, *self.shape)
        else:
            return x.view(*self.shape)


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )
