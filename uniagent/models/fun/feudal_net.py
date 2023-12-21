"""
This file implements a FeuDal net refer to: https://github.com/vtalpaert/pytorch-feudal-network/blob/master/fun.py
"""

from typing import Tuple, List, Any, Dict
from collections import namedtuple

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from gym import spaces

from .utils import reset_grad2, cosine_similarity
from .perception import Perception
from .manager import Manager
from .worker import Worker


FeudalState = namedtuple(
    "FeudalState", "manager_state,worker_state,state_seg, goal_seg"
)


class FeudalNet(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        d: int = 256,
        k: int = 16,
        c: int = 10,
        channel_first: bool = True,
        device: torch.DeviceObjType = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.d, self.k, self.c = d, k, c
        self.device = device

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
        elif action_space.__class__.__name__ == "Box":
            raise NotImplementedError
            # we first test this code with a softmax at the end
            # num_outputs = action_space.shape[0]
        elif isinstance(action_space, int):
            num_outputs = action_space
        else:
            raise NotImplementedError

        self.num_outputs = num_outputs
        self.observation_space = observation_space
        self.action_space = action_space
        self.channel_first = channel_first
        self.perception = self.create_perception()
        self.worker = self.create_worker()
        self.manager = self.create_manager()
        # self.manager_partial_loss = nn.CosineEmbeddingLoss()
        self.to(self.device)

    def create_worker(self) -> Worker:
        return Worker(self.num_outputs, self.d, self.k, device=self.device)

    def create_manager(self) -> Manager:
        return Manager(self.d, self.c, device=self.device)

    def create_perception(self) -> nn.Module:
        perception = Perception(
            self.observation_space.shape, self.d, self.channel_first
        )
        perception.to(self.device)
        perception.device = self.device
        return perception

    def init_weights(self):
        """all submodules are already initialized like this"""

        def default_init(m):
            """Default is a uniform distribution"""
            for module_type in [nn.Linear, nn.Conv2d, nn.LSTMCell]:
                if isinstance(m, module_type):
                    m.reset_parameters()

        self.apply(default_init)

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Any, feudal_state: FeudalState, reset_value_grad: bool = True):
        """Feed forward computing with given observation (x) and feudal state (feudal_state).

        Args:
            x (Any): A batch of observations
            feudal_state (FeudalState): An instance of FeudalState
            reset_value_grad (bool, optional): Whether reset value grad or not. Defaults to False.

        Returns:
            Tuple: ...
        """

        states_W, states_M, ss, goal_hist = (
            feudal_state.worker_state,
            feudal_state.manager_state,
            feudal_state.state_seg,
            feudal_state.goal_seg,
        )
        # detach states of Manager
        tick_dlstm, hx_M, cx_M = states_M

        s_prev = ss[tick_dlstm]
        g_prev = goal_hist[tick_dlstm]

        z = self.perception(x)
        value_manager, goal, s, states_M = self.manager(z, states_M, reset_value_grad)

        assert not s.requires_grad, "s should not require grad"

        # update goal_t and state_t, by replacing goal_{t-c} and state_{t-c}
        ss[tick_dlstm] = s
        goal_hist[tick_dlstm] = goal

        assert s_prev.shape == g_prev.shape, (s_prev.shape, g_prev.shape)
        dcos_t_minus_c = cosine_similarity((s - s_prev), g_prev, keepdims=False)

        sum_goal = sum(map(F.normalize, states_M[2]))
        value_worker, action_probs, states_W = self.worker(
            z, sum_goal.detach(), states_W, reset_value_grad
        )

        return (
            value_worker,
            value_manager,
            action_probs,
            goal,
            dcos_t_minus_c,
            FeudalState(states_M, states_W, ss, goal_hist),
        )

    def init_state(self, batch_size: int) -> FeudalState:
        """Initialize network state, and return a tuple of worker states, manager states and state pair.

        Args:
            batch_size (int): Batch size.

        Returns:
            FeudalState: A tuple of worker states, manager states and state pair.
        """

        # state seg is a list of len=c, each element is a tensor of size [batch x d]
        ss = [
            torch.zeros(batch_size, self.d, requires_grad=False, device=self.device)
            for _ in range(self.c)
        ]
        goals = [
            torch.zeros(batch_size, self.d, requires_grad=False, device=self.device)
            for _ in range(self.c)
        ]
        return FeudalState(
            self.manager.init_state(batch_size),
            self.worker.init_state(batch_size),
            ss,
            goals,
        )

    def reset_states_grad(self, feudal_state: FeudalState) -> FeudalState:
        return FeudalState(
            self.manager.reset_states_grad(feudal_state.manager_state),
            self.worker.reset_states_grad(feudal_state.worker_state),
            list(map(lambda x: reset_grad2(x, False), feudal_state.state_seg)),
            list(map(lambda x: reset_grad2(x, False), feudal_state.goal_seg)),
        )

    def intrinsic_reward(
        self,
        obs: np.ndarray,
        reward: float,
        env_info: Dict[str, Any],
        feudal_state: FeudalState,
    ):
        # states_W, states_M, ss = states
        tick, hx_M, cx_M = feudal_state.manager_state
        t = (tick - 1) % self.c  # since tick is always ahead
        s_t = feudal_state.state_seg[t]
        rI = torch.zeros(s_t.size(0), 1, device=s_t.device)

        for i in range(1, self.c):
            t_minus_i = (t - i) % self.c
            s_t_i = feudal_state.state_seg[t_minus_i]
            g_t_i = F.normalize(cx_M[t_minus_i].data)
            rI += cosine_similarity(
                s_t - s_t_i, g_t_i, keepdims=True
            ).detach()  # F.cosine_similarity(s_t - s_t_i, g_t_i)
        return rI / self.c


def test_forward():
    from gym.spaces import Box
    import numpy as np

    batch = 4
    action_space = 6
    height = 128
    width = 128
    observation_space = Box(0, 255, [3, height, width], dtype=np.uint8)
    fun = FeudalNet(observation_space, action_space, channel_first=True)
    states = fun.init_state(batch)

    for i in range(10):
        image_batch = torch.randn(batch, 3, height, width, requires_grad=True)
        value_worker, value_manager, action_probs, goal, nabla_dcos, states = fun(
            image_batch, states
        )
        print("value worker", value_worker, "value manager", value_manager)
        print("intrinsic reward", fun._intrinsic_reward(states))


if __name__ == "__main__":
    test_forward()
