from collections import defaultdict
from gym.wrappers.record_video import RecordVideo

import tree
import numpy as np

from envs.custom_mujoco.ant_env import AntEnv
from envs.evaluation import get_option_colors


env = AntEnv(render_hw=100)

trajectories = []
for _ in range(5):
    info_steps = defaultdict(lambda: list())
    _, info = env.reset(return_info=True)
    done = False
    for k, v in info.items():
        info_steps[k].append(v)
    while not done:
        _, _, done, info = env.step(env.action_space.sample())
        for k, v in info.items():
            info_steps[k].append(v)
    trajectories.append({"env_infos": {k: np.stack(v) for k, v in info_steps.items()}})


from matplotlib import figure

options = np.random.random((5, 2)) * 8 - 4
colors = get_option_colors(options)
fig = figure.Figure()
ax = fig.add_subplot()
env.render_trajectories(trajectories, colors, plot_axis=None, ax=ax)
fig.savefig("./test.png")
