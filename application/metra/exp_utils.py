class ExpManager:
    def __init__(self) -> None:
        self._training_cnt = 0

    @property
    def snapshotter(self):
        raise NotImplementedError

    @property
    def step_itr(self) -> int:
        """Tracking training step"""
        return self._training_cnt

    @property
    def eval_plot_axis(self):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        loss_info = self.update(*args, **kwargs)
