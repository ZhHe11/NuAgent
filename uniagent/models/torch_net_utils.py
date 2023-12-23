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
            print(
                "xxxxxxxxxxxxx",
                x.shape,
                x.size(0),
                self.shape,
                x.reshape(-1, *self.shape).shape,
            )
            return x.view(-1, *self.shape)
        else:
            return x.view(*self.shape)
