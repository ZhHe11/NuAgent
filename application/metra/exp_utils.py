import os

from uniagent.utils.snapshotter import Snapshotter


class ExpManager:
    def __init__(
        self,
        snapshot_dir=os.path.join(os.getcwd(), "data/local/experiment"),
        snapshot_mode="last",
        snapshot_gap=1,
    ) -> None:
        self._training_cnt = 0
        self._snapshooter = Snapshotter(snapshot_dir, snapshot_mode, snapshot_gap)

    @property
    def snapshotter(self):
        return self._snapshooter

    @property
    def step_itr(self) -> int:
        """Tracking training step"""
        return self._training_cnt

    @property
    def eval_plot_axis(self):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        loss_info = self.update(*args, **kwargs)
        return loss_info
