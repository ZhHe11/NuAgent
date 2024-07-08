import d4rl
import gym
import numpy as np
import functools as ft

def get_inner_env(env):
    if hasattr(env, '_maze_size_scaling'):
        return env
    elif hasattr(env, 'env'):
        return get_inner_env(env.env)
    elif hasattr(env, 'wrapped_env'):
        return get_inner_env(env.wrapped_env)
    return env

def valid_goal_sampler(self, np_random):
    valid_cells = []
    goal_cells = []
    # print('Hello')

    for i in range(len(self._maze_map)):
      for j in range(len(self._maze_map[0])):
        if self._maze_map[i][j] in [0, 'r', 'g']:
          valid_cells.append((i, j))

    # If there is a 'goal' designated, use that. Otherwise, any valid cell can
    # be a goal.
    sample_choices = valid_cells
    cell = sample_choices[np_random.choice(len(sample_choices))]
    xy = self._rowcol_to_xy(cell, add_random_noise=True)

    random_x = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling
    random_y = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling

    xy = (max(xy[0] + random_x, 0), max(xy[1] + random_y, 0))

    return xy

class MazeWrapper(gym.Wrapper):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.render(mode='rgb_array', width=200, height=200)
        self.env_name = env_name
        self.inner_env = get_inner_env(self.env)
        if 'antmaze' in env_name:
            if 'medium' in env_name:
                self.env.viewer.cam.lookat[0] = 10
                self.env.viewer.cam.lookat[1] = 10
                self.env.viewer.cam.distance = 40
                self.env.viewer.cam.elevation = -90
            elif 'umaze' in env_name:
                self.env.viewer.cam.lookat[0] = 4
                self.env.viewer.cam.lookat[1] = 4
                self.env.viewer.cam.distance = 30
                self.env.viewer.cam.elevation = -90
            elif 'large' in env_name:
                self.env.viewer.cam.lookat[0] = 18
                self.env.viewer.cam.lookat[1] = 13
                self.env.viewer.cam.distance = 55
                self.env.viewer.cam.elevation = -90
            self.inner_env.goal_sampler = ft.partial(valid_goal_sampler, self.inner_env)
        elif 'maze2d' in env_name:
            if 'open' in env_name:
                pass
            elif 'large' in env_name:
                self.env.viewer.cam.lookat[0] = 5
                self.env.viewer.cam.lookat[1] = 6.5
                self.env.viewer.cam.distance = 15
                self.env.viewer.cam.elevation = -90
                self.env.viewer.cam.azimuth = 180
            self.draw_ant_maze = get_inner_env(gym.make('antmaze-large-diverse-v2'))
        self.action_space = self.env.action_space

    def render(self, *args, **kwargs):
        img = self.env.render(*args, **kwargs)
        if 'maze2d' in self.env_name:
            img = img[::-1]
        return img



