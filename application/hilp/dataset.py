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
