from typing import List, Tuple, Sequence

import random
import numpy as np


class SimpleReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int = None, indices: Sequence[int] = None
    ) -> List[Tuple]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


import torch


class ReplayArray:
    def __init__(self, capacity: int, inner_shape: Sequence[int], dtype: np.dtype):
        self.capacity = capacity
        self.inner_shape = inner_shape
        self.dtype = dtype
        self.data = np.empty((capacity,) + tuple(inner_shape), dtype=dtype)

        self.flag = -1
        self.size = 0

        self.reset()

    def reset(self):
        self.flag = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def batch_push(
        self,
        batched_item_: np.ndarray,
        use_copy: bool = True,
        update_signal: bool = False,
    ) -> Tuple[int, int]:
        """Push a batch of data, but do not update flag and size by default

        Args:
            batched_item_ (np.ndarray): A batch of data
            use_copy (bool, optional): Enable copy or not. Defaults to True.

        Returns:
            Tuple[int, int]: A tuple of latest flag and real buffer size.
        """

        flag = self.flag
        if use_copy:
            batched_item_ = batched_item_.copy()
        batch_size = batched_item_.shape[0]

        untruncated_length = flag + batch_size
        repeat_num = untruncated_length // self.capacity
        tail_len = untruncated_length - repeat_num * self.capacity
        if repeat_num:
            flag = flag + tail_len
            data = batched_item_[-self.capacity :]
            data = np.roll(data, shift=-flag)
            self.data[:] = data
        else:
            tail_len_in_buffer = self.capacity - flag
            self.data[flag:] = batched_item_[:tail_len_in_buffer]
            head_len_in_buffer = batch_size - tail_len_in_buffer
            self.data[:head_len_in_buffer] = batched_item_[tail_len_in_buffer:]
            flag = head_len_in_buffer

        cur_size = min(self.size + untruncated_length, self.capacity)
        if update_signal:
            self.flag = flag
            self.size = cur_size

        return flag, cur_size

    def push(
        self, item_: np.ndarray, batch_flatten: bool = False, use_copy: bool = True
    ) -> Tuple[int, int]:
        """Return latest flag and buffer size.

        Args:
            item_ (np.ndarray): Given inserted data
            batch_flatten (bool, optional): Whether flat the given batch. Defaults to False.
            use_copy (bool, optional): Copy data or not. Defaults to True.

        Returns:
            Tuple[int, int]: A tuple of latest flag and real buffer size
        """

        flag = self.flag
        if use_copy:
            item_ = item_.copy()

        assert (
            item_.dtype == self.dtype
        ), f"expected dtype is: {self.dtype} while got {item_.dtype}"
        if item_.shape == self.inner_shape:
            self.data[flag] = item_
            flag = 0 if flag + 1 == self.capacity else flag + 1
            cur_size = min(self.size + 1, self.capacity)
        elif len(item_.shape) > len(self.inner_shape):
            if batch_flatten:
                item_ = item_.resahpe((-1,) + self.inner_shape)
            else:
                assert (
                    len(item_.shape) == len(item_.shape) + 1
                ), f"the batch dim should be 1-D, otherwise set batch_flatten to True"
            flag, cur_size = self.batch_push(item_, use_copy=False, update_signal=False)

        self.flag = flag
        self.size = cur_size

        return self.flag, self.size

    def sample(
        self,
        indices: Sequence[int] = None,
        batch_size: int = None,
        to_torch: bool = False,
        device: str = "cpu",
    ):
        if batch_size is not None and batch_size > 0:
            indices = np.random.choice(self.size, batch_size)
        if to_torch:
            return torch.from_numpy(self.data[indices]).to(device)
        else:
            return self.data[indices].copy()


from typing import Dict, Union

import tree


class ReplayBuffer:
    def __init__(
        self, capacity: int, shape_and_dtypes: Dict[str, Tuple[Sequence[int], np.dtype]]
    ):
        self.data = {k: ReplayArray(capacity, *v) for k, v in shape_and_dtypes.items()}
        self.meta_item = list(self.data.values())[0]
        self.capacity = capacity

    def __len__(self):
        return len(self.meta_item)

    def push(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].push(v)

    def sample_indices(self, batch_size: int):
        return np.random.choice(len(self), batch_size)

    def sample(
        self,
        batch_size: int = None,
        indices: Sequence[int] = None,
        to_torch: bool = False,
        device: str = "cpu",
    ):
        if batch_size is not None:
            indices = self.sample_indices(batch_size)
        batch = tree.map_structure(
            lambda x: x.sample(indices=indices, to_torch=to_torch, device=device),
            self.data,
        )
        return batch

    def sample_in_continuous(
        self,
        start: int,
        seg_length: int,
        truncated: bool = False,
        outer_indices_strategy: str = "repeat",
        to_torch: bool = False,
        device: str = "cpu",
    ) -> Union[torch.Tensor, np.ndarray]:
        """Sample a batch with continuous indices. Especially for trajectory-level sample.

        Args:
            start (int): The start index
            seg_length (int): Expected batch_size, if truncated is set to True, the length of returned data may be less than it
            truncated (bool, optional): Whether trucated outer indices, if not, then the outer indices will be determined by strategy. Defaults to False.
            outer_indices_strategy (str, optional): The strategy for outer indices, 'repeat' for use repeated last indices, 'roll' for rolling. Defaults to 'repeat'
            to_torch (bool, optional): _description_. Defaults to False.
            device (str, optional): _description_. Defaults to "cpu".

        Returns:
            Union[torch.Tensor, np.ndarray]: _description_
        """
        size = len(self)
        if start + seg_length > size:
            new_seg_length = size - start
            if truncated:
                seg_length = new_seg_length
                indices = start + np.arange(seg_length)
            else:
                if outer_indices_strategy == "repeat":
                    indices = (start + np.arange(new_seg_length)).tolist()
                    indices = indices + [indices[-1]] * (seg_length - new_seg_length)
                    indices = np.asarray(indices)
                elif outer_indices_strategy == "roll":
                    indices = np.roll(np.arange(size), start)[:seg_length]
                else:
                    raise RuntimeError(
                        f"unexpected outer indices strategy: {outer_indices_strategy}"
                    )
        else:
            indices = start + np.arange(seg_length)

        return self.sample(indices=indices, to_torch=to_torch, device=device)
