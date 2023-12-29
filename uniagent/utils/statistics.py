from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class RunningMeanStd:
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    :param mean: the initial mean estimation for data array. Default to 0.
    :param std: the initial standard error estimation for data array. Default to 1.
    :param clip_max: the maximum absolute value for data array. Default to
        10.0.
    :param epsilon: To avoid division by zero.
    """

    def __init__(
        self,
        mean: Union[float, np.ndarray, torch.Tensor] = 0.0,
        std: Union[float, np.ndarray, torch.Tensor] = 1.0,
        clip_max: Union[float, None] = 10.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:
        self.mean, self.var = mean, std
        self.clip_max = clip_max
        self.count = 0
        self.eps = epsilon

    @torch.no_grad()
    def norm(
        self, data_array: Union[float, np.ndarray, torch.Tensor]
    ) -> Union[float, np.ndarray, torch.Tensor]:
        if isinstance(data_array, torch.Tensor):
            with torch.no_grad():
                data_array = (data_array - self.mean) / torch.sqrt(self.var + self.eps)
        else:
            data_array = (data_array - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_max:
            data_array = np.clip(data_array, -self.clip_max, self.clip_max)
        return data_array

    @torch.no_grad()
    def update(self, data_array: Union[np.ndarray, torch.Tensor]) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""

        if isinstance(data_array, torch.Tensor):
            batch_mean, batch_var = torch.mean(data_array, dim=0), torch.var(
                data_array, dim=0
            )
        else:
            batch_mean, batch_var = np.mean(data_array, axis=0), np.var(
                data_array, axis=0
            )
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count
