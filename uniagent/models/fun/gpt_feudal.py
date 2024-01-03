from typing import Tuple, List, Any, Dict
from collections import namedtuple, deque
from argparse import Namespace

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
        self, observation_space: Space, action_space: Space, args: Namespace
    ) -> None:
        super().__init__(observation_space, action_space, args)

    def create_worker(self) -> Worker:
        return TransformerWorker(self.observation_space, self.action_space, self.config)

    def create_manager(self) -> Manager:
        return TransformerManager(
            self.observation_space, self.action_space, self.config
        )

    def init_memory(self):
        raise NotImplementedError

    def init_weights(self):
        pass
