from typing import Tuple, List, Any, Dict
from collections import namedtuple, deque

import torch
import numpy as np

from torch import DeviceObjType, nn
from torch._C import DeviceObjType, device
from torch.nn import functional as F
from gym import Space, spaces

from .utils import reset_grad2, cosine_similarity
from .perception import Perception
from .manager import TransformerManager, Manager
from .worker import TransformerWorker, Worker
from .feudal_net import FeudalState, FeudalNet


class GPTFeudal(FeudalNet):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        backbone: str = "gpt2",
        d: int = 256,
        k: int = 16,
        c: int = 10,
        channel_first: bool = True,
        device: DeviceObjType = ...,
    ) -> None:
        self.backbone = backbone
        super().__init__(
            observation_space, action_space, d, k, c, channel_first, device
        )

    def create_worker(self) -> Worker:
        return TransformerWorker(
            self.backbone, self.num_outputs, self.d, self.k, self.backbone, self.device
        )

    def create_manager(self) -> Manager:
        return TransformerManager(self.backbone, self.d, self.k, self.c, self.device)

    def init_weights(self):
        pass
