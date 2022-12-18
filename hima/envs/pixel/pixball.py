#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np


class Pixball:
    def __init__(
            self,
            size,
            initial_pos,
            forces,
            seed: int = None
    ):
        self.size = np.array(size, dtype=np.int)
        self.env_state = np.zeros(size)
        self.force_pos = np.full(size, fill_value=-1)
        self.initial_pos = np.array(initial_pos, dtype=np.int)
        self.current_pos = self.initial_pos.copy()
        self.forces = forces

        self.velocity = np.zeros(2, dtype=np.int)
        self.env_state[initial_pos[0], initial_pos[1]] = 1

        for i, force in enumerate(self.forces):
            pos = force['pos']
            self.force_pos[pos] = i

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, velocity):
        self.velocity = np.array(velocity, dtype=np.int)

    def obs(self):
        return self.env_state

    def step(self):
        self.env_state[self.current_pos[0], self.current_pos[1]] = 0
        force = self.force_pos[self.current_pos[0], self.current_pos[1]]

        if force != -1:
            directions = self.forces[force]['directions']
            delta = self._rng.choice(directions)
            self.velocity += delta

        new_pos = self.current_pos + self.velocity

        for i in range(len(self.size)):
            if new_pos[i] <= 0 or new_pos[i] >= (self.size[i]-1):
                self.velocity[i] = - self.velocity[i]

        self.current_pos = np.clip(new_pos, a_min=0, a_max=self.size-1)

        self.env_state[self.current_pos[0], self.current_pos[1]] = 1

    def reset(self):
        self.env_state = np.zeros(self.size)
        self.current_pos = self.initial_pos.copy()
        self.env_state[self.current_pos[0], self.current_pos[1]] = 1
        self.velocity = np.zeros(2, dtype=np.int)
