from torch import nn
from .utils import View


class Perception(nn.Module):
    """Returns z, the shared intermediate representation [batch x d]"""

    def __init__(self, observation_shape, d, channel_first):
        super(Perception, self).__init__()
        self.channel_first = channel_first

        # Note that we expect input in Pytorch images' style : batch x C x H x W (this is arg channel_first)
        # but in gym a Box.shape for an image is (H, W, C) (use channel_first=False)
        if channel_first:
            channels, height, width = observation_shape
        else:
            height, width, channels = observation_shape
            self.view = View((channels, height, width))

        percept_linear_in = (
            32
            * int((int((height - 4) / 4) - 2) / 2)
            * int((int((width - 4) / 4) - 2) / 2)
        )
        self.f_percept = nn.Sequential(
            nn.Conv2d(channels, 16, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=2),
            nn.ReLU(),
            View((percept_linear_in,)),
            nn.Linear(percept_linear_in, d),
            nn.ReLU(),
        )

    def forward(self, x):
        if not self.channel_first:
            x = self.view(x)
        return self.f_percept(x)
