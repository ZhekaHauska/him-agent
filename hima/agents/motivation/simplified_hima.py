#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.common.sdr import SparseSdr
from hima.common.config.base import TConfig
from hima.modules.simple_bg import BasalGanglia


class Agent:
    def __init__(self, seed: int, state_dim: int, n_actions: int, config: TConfig):

        self.action_patterns = []
        b = config['action_bucket_size']
        for a in range(n_actions):
            self.action_patterns.append(np.arange(a*b, (a+1)*b))
        action_dim = n_actions * b

        self.bg = BasalGanglia(seed, state_dim, action_dim, **config['bg'])

    def act(self, state: SparseSdr, reward: float, is_first: bool) -> int:
        if is_first:
            self.bg.reset()

        action = self.bg.compute(state, self.action_patterns)
        self.bg.update(reward)
        return action
