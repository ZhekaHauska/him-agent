#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from copy import copy


class GridWorld:
    def __init__(
            self,
            room,
            default_reward=0,
            observation_radius=0,
            collision_hint=False,
            seed=None,
    ):
        self._rng = np.random.default_rng(seed)

        self.colors, self.rewards, self.terminals = (
            room[0, :, :], room[1, :, :], room[2, :, :]
        )

        self.h, self.w = self.colors.shape

        self.return_state = observation_radius < 0
        self.observation_radius = observation_radius
        self.collision_hint = collision_hint

        self.shift = max(self.observation_radius, 1)

        self.colors = np.pad(
            self.colors,
            self.shift,
            mode='constant',
            constant_values=-1
        )

        self.n_colors = len(np.unique(self.colors))

        if not self.return_state:
            self.observation_shape = (2*self.observation_radius + 1, 2*self.observation_radius + 1)
        else:
            self.observation_shape = (2,)

        self.start_r = None
        self.start_c = None
        self.r = None
        self.c = None
        self.action = None
        self.temp_obs = None
        # left, right, up, down
        self.actions = {0, 1, 2, 3}
        self.default_reward = default_reward

    def reset(self, start_r=None, start_c=None):
        if start_r is None:
            start_r = self._rng.integers(self.h)
        if start_c is None:
            start_c = self._rng.integers(self.w)

        self.start_r, self.start_c = start_r, start_c
        self.r, self.c = start_r, start_c
        self.temp_obs = None

    def obs(self):
        assert self.r is not None
        assert self.c is not None
        if self.return_state:
            obs = (self.r, self.c)
        else:
            if self.temp_obs is not None:
                obs = copy(self.temp_obs)
                self.temp_obs = None
            else:
                obs = self._get_obs(self.r, self.c)
        return (
            obs,
            self.rewards[self.r, self.c] + self.default_reward,
            bool(self.terminals[self.r, self.c])
        )

    def act(self, action):
        assert action in self.actions
        self.action = action

    def step(self):
        if self.action is not None:
            assert self.r is not None
            assert self.c is not None

            prev_r = self.r
            prev_c = self.c

            if self.action == 0:
                self.c -= 1
            elif self.action == 1:
                self.c += 1
            elif self.action == 2:
                self.r -= 1
            elif self.action == 3:
                self.r += 1

            # Check whether action is taking to inaccessible states.
            temp_x = self.colors[self.r+self.shift, self.c+self.shift]
            if temp_x < 0:
                self.r = prev_r
                self.c = prev_c

                if (not self.return_state) and self.collision_hint:
                    self.temp_obs = np.full(self.observation_shape, fill_value=temp_x)

    def _get_obs(self, r, c):
        r += self.shift
        c += self.shift
        start_r, start_c = r - self.observation_radius, c - self.observation_radius
        end_r, end_c = r + self.observation_radius + 1, c + self.observation_radius + 1
        obs = self.colors[start_r:end_r, start_c:end_c]
        return obs
