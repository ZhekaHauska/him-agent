#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.experiments.successor_representations.runners.base import BaseEnvironment
import os
import numpy as np
from hima.common.config.base import read_config
import io


class PinballWrapper(BaseEnvironment):
    def __init__(self, conf, setup):
        from pinball import Pinball
        self.actions = conf.pop('actions')
        if 'start_position' in conf.keys():
            self.start_position = conf.pop('start_position')
        else:
            self.start_position = None

        conf['exe_path'] = self._get_exe_path()
        conf['config_path'] = self._get_setup_path(setup)
        self.environment = Pinball(**conf)
        obs, _, _ = self.environment.obs()
        self.raw_obs_shape = (obs.shape[0], obs.shape[1])
        self.n_actions = len(self.actions)

    def obs(self):
        return self.environment.obs()

    def act(self, action):
        if action is not None:
            pinball_action = self.actions[action]
            return self.environment.act(pinball_action)

    def step(self):
        return self.environment.step()

    def reset(self):
        return self.environment.reset(self.start_position)

    def change_setup(self, setup):
        self.environment.set_config(
            self._get_setup_path(setup)
        )

    def close(self):
        return self.environment.close()

    @staticmethod
    def _get_setup_path(setup):
        return os.path.join(
            os.environ.get('PINBALL_ROOT', None),
            'configs',
            f"{setup}.json"
        )

    @staticmethod
    def _get_exe_path():
        return os.environ.get('PINBALL_EXE', None)


class AnimalAIWrapper(BaseEnvironment):
    def __init__(self, conf, setup):
        from animalai.actions import AAIActions

        conf['file_name'] = self._get_exe_path()
        self.conf = conf
        self.setup = setup
        self.environment, self.behavior = self._start_env(self.setup)

        self.raw_obs_shape = self.environment.behavior_specs[self.behavior].observation_specs[
            0].shape[:2]

        self.actions = (
            AAIActions().LEFT,
            AAIActions().FORWARDS,
            AAIActions().RIGHT,
            # AAIActions().BACKWARDS
        )
        self.n_actions = len(self.actions)

    def obs(self):
        dec, term = self.environment.get_steps(self.behavior)

        obs = None
        reward = 0
        is_terminal = False

        if len(dec) > 0:
            obs = self.environment.get_obs_dict(dec.obs)["camera"]
            reward += dec.reward

        if len(term):
            obs = self.environment.get_obs_dict(term.obs)["camera"]
            reward += term.reward
            is_terminal = True

        return obs, reward, is_terminal

    def act(self, action):
        if action is not None:
            aai_action = self.actions[action]
            self.environment.set_actions(self.behavior, aai_action.action_tuple)

    def step(self):
        self.environment.step()

    def reset(self):
        self.environment.reset(self._get_setup_path(self.setup))

    def change_setup(self, setup):
        self.environment.close()
        self.setup = setup
        self.environment, self.behavior = self._start_env(self.setup)

    def close(self):
        self.environment.close()

    def _start_env(self, setup):
        from animalai.environment import AnimalAIEnvironment
        from mlagents_envs.exception import UnityWorkerInUseException

        worker_id = 0
        self.conf['arenas_configurations'] = self._get_setup_path(setup)
        while True:
            try:
                environment = AnimalAIEnvironment(
                    worker_id=worker_id,
                    **self.conf
                )
                break
            except UnityWorkerInUseException:
                worker_id += 1

        behavior = list(environment.behavior_specs.keys())[0]
        return environment, behavior

    @staticmethod
    def _get_setup_path(setup):
        return os.path.join(
            os.environ.get('ANIMALAI_ROOT', None),
            'configs',
            f"{setup}"
        )

    @staticmethod
    def _get_exe_path():
        return os.environ.get('ANIMALAI_EXE', None)


class GridWorldWrapper(BaseEnvironment):
    def __init__(self, conf, setup):
        if 'start_position' in conf:
            self.start_position = conf.pop('start_position')
        else:
            self.start_position = (None, None)

        self.conf = conf
        self.environment = self._start_env(setup)
        self.n_colors = self.environment.n_colors
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
                self.n_colors
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

    @property
    def current_state(self):
        return self.environment.c + self.environment.r*self.environment.w

    @property
    def render(self):
        import matplotlib.pyplot as plt
        from PIL import Image
        shift = self.environment.shift
        im = self.environment.colors.copy()
        agent_color = max(self.environment.unique_colors) + 0.5

        if shift > 0:
            im = im[shift:-shift, shift:-shift]

        im[self.environment.r, self.environment.c] = agent_color

        plt.figure()
        plt.imshow(im, cmap='Pastel1', aspect=1, vmin=self.min_vis_color)
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
        from hima.envs.gridworld import GridWorld
        config = read_config(
            self._get_setup_path(setup)
        )

        env = GridWorld(
                room=np.array(config['room']),
                default_reward=config['default_reward'],
                **self.conf
        )

        return env

    @staticmethod
    def _get_setup_path(setup):
        return os.path.join(
                os.environ.get('GRIDWORLD_ROOT', None),
                f"{setup}.yaml"
            )
