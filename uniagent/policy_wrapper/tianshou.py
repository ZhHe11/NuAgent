from typing import Callable

import torch


def policy_func_wrapper(policy) -> Callable[[torch.Tensor], torch.Tensor]:
    def f(obs_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    return f
