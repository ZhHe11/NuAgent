import torch

from torch import nn


class DilatedLSTM(nn.Module):
    def __init__(self, r: int, input_size: int, hidden_size: int) -> None:
        super().__init__()
