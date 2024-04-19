import copy
import pathlib
import time
import os

import numpy as np
import torch
import platform

if "macOS" in platform.platform():
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

from moviepy import editor as mpy
from matplotlib import figure
from matplotlib.patches import Ellipse
from sklearn import decomposition


def draw_2d_gaussians(
    means,
    stddevs,
    colors,
    ax,
    fill=False,
    alpha=0.8,
    use_adaptive_axis=False,
    draw_unit_gaussian=True,
    plot_axis=None,
):
    means = np.clip(means, -1000, 1000)
    stddevs = np.clip(stddevs, -1000, 1000)
    square_axis_limit = 2.0
    if draw_unit_gaussian:
        ellipse = Ellipse(
            xy=(0, 0),
            width=2,
            height=2,
            edgecolor="r",
            lw=1,
            facecolor="none",
            alpha=0.5,
        )
        ax.add_patch(ellipse)
    for mean, stddev, color in zip(means, stddevs, colors):
        if len(mean) == 1:
            mean = np.concatenate([mean, [0.0]])
            stddev = np.concatenate([stddev, [0.1]])
        ellipse = Ellipse(
            xy=mean,
            width=stddev[0] * 2,
            height=stddev[1] * 2,
            edgecolor=color,
            lw=1,
            facecolor="none" if not fill else color,
            alpha=alpha,
        )
        ax.add_patch(ellipse)
        square_axis_limit = max(
            square_axis_limit,
            np.abs(mean[0] + stddev[0]),
            np.abs(mean[0] - stddev[0]),
            np.abs(mean[1] + stddev[1]),
            np.abs(mean[1] - stddev[1]),
        )
    square_axis_limit = square_axis_limit * 1.2
    ax.axis("scaled")
    if plot_axis is None:
        if use_adaptive_axis:
            ax.set_xlim(-square_axis_limit, square_axis_limit)
            ax.set_ylim(-square_axis_limit, square_axis_limit)
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
    else:
        ax.axis(plot_axis)


def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None,]

    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.0

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if n_cols is None:
        if v.shape[0] <= 3:
            n_cols = v.shape[0]
        elif v.shape[0] <= 9:
            n_cols = 3
        else:
            n_cols = 6
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def save_video(runner, label, tensor, fps=15, n_cols=None):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t

    if tensor.dtype in [object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    # Encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    plot_path = (
        pathlib.Path(runner._snapshotter.snapshot_dir)
        / "plots"
        # / f'{label}_{runner.step_itr}.gif')
        / f"{label}_{runner.step_itr}.mp4"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    clip.write_videofile(str(plot_path), audio=False, verbose=False, logger=None)
    if "WANDB_API_KEY" in os.environ:
        import wandb

        wandb.log({label: wandb.Video(str(plot_path))}, step=runner.step_itr)


def record_video(runner, label, trajectories, n_cols=None, skip_frames=1):
    renders = []
    for trajectory in trajectories:
        render = trajectory["env_infos"]["render"]
        if render.ndim >= 5:
            render = render.reshape(-1, *render.shape[-3:])
        elif render.ndim == 1:
            render = np.concatenate(render, axis=0)
        renders.append(render)
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        renders[i] = np.concatenate(
            [
                render,
                np.zeros(
                    (max_length - render.shape[0], *render.shape[1:]),
                    dtype=render.dtype,
                ),
            ],
            axis=0,
        )
        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders)
    save_video(runner, label, renders, n_cols=n_cols)


from torch import nn


class FigManager:
    def __init__(self, agent: nn.Module, label, extensions=None, subplot_spec=None):
        self.agent = agent
        self.label = label
        self.fig = figure.Figure()
        if subplot_spec is not None:
            self.ax = self.fig.subplots(*subplot_spec).flatten()
        else:
            self.ax = self.fig.add_subplot()

        if extensions is None:
            self.extensions = ["png"]
        else:
            self.extensions = extensions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plot_paths = [
            (
                pathlib.Path(self.agent.snapshotter.snapshot_dir)
                / "plots"
                / f"{self.label}_{self.agent.step_itr}.{extension}"
            )
            for extension in self.extensions
        ]
        plot_paths[0].parent.mkdir(parents=True, exist_ok=True)
        for plot_path in plot_paths:
            self.fig.savefig(plot_path, dpi=300)
