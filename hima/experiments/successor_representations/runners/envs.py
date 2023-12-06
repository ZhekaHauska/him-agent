#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.experiments.successor_representations.runners.base import BaseEnvironment
from pinball import Pinball
import os


class PinballWrapper(BaseEnvironment):
    environment: Pinball

    def __init__(self, conf):
        self.actions = conf.pop('actions')
        if 'start_position' in conf.keys():
            self.start_position = conf.pop('start_position')
        else:
            self.start_position = None

        conf['exe_path'] = self._get_exe_path()
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
        return self

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
