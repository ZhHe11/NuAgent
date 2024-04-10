from typing import Dict

import tree
import numpy as np


def get_size(data) -> int:
    sizes = tree.map_structure(lambda arr: len(arr), data)
    return max(tree.flatten(sizes))


class Dataset:
    """
    A class for storing (and retrieving batches of) data in nested dictionary format.

    Example:
        dataset = Dataset({
            'observations': {
                'image': np.random.randn(100, 28, 28, 1),
                'state': np.random.randn(100, 4),
            },
            'actions': np.random.randn(100, 2),
        })

        batch = dataset.sample(32)
        # Batch should have nested shape: {
        # 'observations': {'image': (32, 28, 28, 1), 'state': (32, 4)},
        # 'actions': (32, 2)
        # }
    """

    @classmethod
    def create(
        cls, observations, actions, rewards, masks, next_observations, **extra_fields
    ):
        data = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "next_observations": next_observations,
            **extra_fields,
        }
        return cls(data)

    def __init__(self, *args, **kwargs):
        self.data = {}
        for arg in args:
            assert isinstance(arg, dict), type(arg)
            self.data.update(arg)
        if kwargs is not None:
            self.data.update(kwargs)
        self.size = get_size(self.data)

    def __getitem__(self, k) -> np.ndarray:
        return self.data[k]

    def copy(self, data: Dict[str, np.ndarray]):
        for k, v in data.items():
            self.data[k] = v
        return self

    def sample(self, batch_size: int, indx=None):
        """
        Sample a batch of data from the dataset. Use `indx` to specify a specific
        set of indices to retrieve. Otherwise, a random sample will be drawn.

        Returns a dictionary with the same structure as the original dataset.
        """
        if indx is None:
            indx = np.random.randint(self.size, size=batch_size)
        return self.get_subset(indx)

    def get_subset(self, indx):
        return tree.map_structure(lambda arr: arr[indx], self.data)


import dataclasses
import jax
import jax.numpy as jnp

from flax.core.frozen_dict import FrozenDict

# from flax.core import freeze


def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


random_crop = jax.jit(random_crop, static_argnames=("padding",))


def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


batched_random_crop = jax.jit(batched_random_crop, static_argnames=("padding",))


@dataclasses.dataclass
class GCDataset:
    """Goal-conditioned dataset, built with existing trajectory dataset"""

    dataset: Dataset
    """offline dataset"""

    p_randomgoal: float
    """probability of random goal sampling"""

    p_trajgoal: float
    """probability of trajectory-level goal sampling"""

    p_currgoal: float
    """probability of current state as goals"""

    discount: float

    geom_sample: int = 1

    terminal_key: str = "dones_float"
    """the human_readable name of terminal key, for sampling"""

    reward_scale: float = 1.0

    reward_shift: float = 0.0

    p_aug: float = None

    def __post_init__(self):
        (self.terminal_locs,) = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)

        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)

        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(
                indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int),
                final_state_indx,
            )
        else:
            middle_goal_indx = np.round(
                (
                    np.minimum(indx + 1, final_state_indx) * distance
                    + final_state_indx * (1 - distance)
                )
            ).astype(int)

        goal_indx = np.where(
            np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal),
            middle_goal_indx,
            goal_indx,
        )

        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(
        self, batch_size: int, indx: np.ndarray = None, evaluation: bool = False
    ) -> Dict[str, jnp.array]:
        """Sample a batch of data for evaluation or traning.

        Args:
            batch_size (int): Batch size.
            indx (np.ndarray, optional): Index array. Defaults to None.
            evaluation (bool, optional): Evaluation mode or not. Defaults to False.

        Returns:
            Dict[str, jnp.array]: A dict of batched data.
        """

        if indx is None:
            indx = np.random.randint(self.dataset.size - 1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = indx == goal_indx

        batch["rewards"] = success.astype(float) * self.reward_scale + self.reward_shift
        batch["masks"] = 1.0 - success.astype(float)
        batch["goals"] = tree.map_structure(
            lambda arr: arr[goal_indx], self.dataset["observations"]
        )

        if self.p_aug is not None and not evaluation:
            if np.random.rand() < self.p_aug:
                aug_keys = ["observations", "next_observations", "goals"]
                padding = 3
                crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
                crop_froms = np.concatenate(
                    [crop_froms, np.zeros((batch_size, 1), dtype=np.int32)], axis=1
                )
                for key in aug_keys:
                    batch[key] = tree.map_structure(
                        lambda arr: np.array(
                            batched_random_crop(arr, crop_froms, padding)
                        )
                        if len(arr.shape) == 4
                        else arr,
                        batch[key],
                    )

        if isinstance(batch["goals"], FrozenDict):
            raise RuntimeError("Unexcepted data type, goals should not be a FrozenDict")

        return batch
