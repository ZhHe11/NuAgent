import torch

from torch import nn


def cosine_similarity(x, y, keepdims: bool = False):
    num = (x * y).sum(-1, keepdims=keepdims)
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    return num / max(1e-6, denom.detach())


def reset_grad2(t, requires_grad=None):
    tr = t.detach()
    tr.requires_grad = t.requires_grad if requires_grad is None else requires_grad
    return tr


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
            return x.view(x.size(0), *self.shape)
        else:
            return x.view(*self.shape)
