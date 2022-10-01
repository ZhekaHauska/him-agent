#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np


class RndAgent:
    n_actions: int

    def __init__(self, n_actions, seed: int):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    @property
    def name(self):
        return 'rnd'

    def act(self):
        return self.rng.integers(self.n_actions)
