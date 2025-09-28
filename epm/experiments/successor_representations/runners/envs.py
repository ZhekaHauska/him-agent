#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from epm.experiments.successor_representations.runners.base import BaseEnvironment
import os
import numpy as np
from epm.common.config.base import read_config
import io


class GridWorldWrapper(BaseEnvironment):
    def __init__(self, conf, setup):
        self.start_position = (None, None)
        self.conf = conf
        self.environment = self._start_env(setup)
        self.n_colors = self.environment.n_colors
        self.max_color = np.max(self.environment.unique_colors)
        self.min_color = np.min(self.environment.unique_colors)
        self.min_vis_color = np.min(self.environment.colors)
        self.trajectory = []
        self.is_first_step = True

        self.n_cells = (
                (self.environment.observation_radius * 2 + 1) ** 2
        )

        if self.environment.return_state:
            self.raw_obs_shape = (1, self.environment.h * self.environment.w)
        else:
            self.raw_obs_shape = (
                self.n_cells,
                self.max_color - self.min_color + 1
            )
        self.actions = tuple(self.environment.actions)
        self.n_actions = len(self.actions)

    def obs(self):
        obs, reward, is_terminal = self.environment.obs()
        if self.environment.return_state:
            obs = [obs[1] + obs[0]*self.environment.w]
        else:
            obs = obs.flatten()
            obs += (
                np.arange(self.n_cells)*self.n_colors - self.min_color
            )

        if self.is_first_step:
            self.trajectory.clear()
        self.trajectory.append(self.state)
        self.is_first_step = False
        return obs, reward, is_terminal

    def act(self, action):
        if action is not None:
            gridworld_action = self.actions[action]
            self.environment.act(gridworld_action)

    def step(self):
        self.environment.step()

    def reset(self):
        self.environment.reset(*self.start_position)
        self.is_first_step = True

    def change_setup(self, setup):
        self.environment = self._start_env(setup)

    def close(self):
        self.environment = None

    def get_true_matrices(self):
        return self.environment.get_true_matrices()

    @property
    def true_state(self):
        return self.environment.c + self.environment.r*self.environment.w

    @property
    def render(self):
        from PIL import Image
        import matplotlib.pyplot as plt
        import seaborn as sns

        im = self.environment.colors - self.environment.unique_colors.min()
        plt.figure()
        sns.heatmap(im, annot=True, cmap='Pastel1', square=True, cbar=False)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches="tight")
        plt.close()
        buf.seek(0)
        im = Image.open(buf)
        return im

    @property
    def state(self):
        shift = self.environment.shift
        im = self.environment.colors.astype('float32')
        agent_color = max(self.environment.unique_colors) + 0.5

        if shift > 0:
            im = im[shift:-shift, shift:-shift]

        im[self.environment.r, self.environment.c] = agent_color
        return im

    def _start_env(self, setup):
        from epm.envs.gridworld import GridWorld
        config = read_config(
            self._get_setup_path(setup)
        )
        if 'start_position' in config:
            self.start_position = config['start_position']
        else:
            self.start_position = (None, None)

        env = GridWorld(
                room=np.array(config['room']),
                **self.conf
        )

        return env

    @staticmethod
    def _get_setup_path(setup):
        return os.path.join(
                os.environ.get('GRIDWORLD_ROOT', None),
                "setups",
                f"{setup}.yaml"
            )
