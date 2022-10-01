#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.utils import isnone
from hima.envs.biogwlab.environment import Environment


class GwAgentStateProvider:
    env: Environment

    def __init__(self, env: Environment):
        self.env = env
        self.origin = None

    @property
    def state(self):
        return self.position, self.view_direction

    @property
    def position(self):
        return self.env.agent.position

    @property
    def view_direction(self):
        return self.env.agent.view_direction

    def overwrite(self, position=None, view_direction=None):
        if self.origin is None:
            self.origin = self.state
        self._set(position, view_direction)

    def restore(self):
        if self.origin is None:
            raise ValueError('Nothing to restore')
        self._set(*self.origin)
        self.origin = None

    def _set(self, position, view_direction):
        self.env.agent.position = isnone(position, self.env.agent.position)
        self.env.agent.view_direction = isnone(view_direction, self.env.agent.view_direction)