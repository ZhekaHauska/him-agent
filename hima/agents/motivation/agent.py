#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config.base import TConfig
import numpy as np
from hima.modules.motivation import Striatum
from hima.common.sdr import SparseSdr
from hima.common.utils import softmax


class Agent:
    def __init__(self, obs_dim: int, action_dim: int, motiv_dim: int, config: TConfig):
        self.seed = config['seed']
        self._rng = np.random.default_rng(self.seed)
        self.temperature = config['temperature']
        self.action_dim = action_dim
        self.motiv_dim = motiv_dim
        self.striatum = Striatum(self.seed, obs_dim, motiv_dim, **config['striatum'])

        self.p = None
        self.obs = None
        self.motiv = None
        self.action = None
        self.full_action = None

    def act(self, obs: SparseSdr, motiv: SparseSdr) -> int:
        self.obs = obs
        self.motiv = motiv
        self.full_action = self.striatum.compute(obs, motiv)
        size = self.striatum.action_size // self.action_dim
        actions = np.array(
            [
                len(np.intersect1d(self.full_action, np.arange(size * i, size * (i + 1))))
                for i in range(self.action_dim)
            ]
        )
        self.p = softmax(actions, self.temperature)
        a = self._rng.choice(self.action_dim, 1, p=self.p)[0]
        self.action = np.intersect1d(self.full_action, np.arange(size * a, size * (a+1)))
        return a

    def update(self, reward: float):
        self.striatum.update(self.obs, self.motiv, self.action, reward)

    def get_value(self) -> float:
        sa = self.striatum.state_weights.xa2sdr(self.obs, self.full_action)
        v = self.striatum.state_weights.dopa_weights.get_value(sa)
        return v

    def get_probs(self) -> np.ndarray:
        return self.p

    def reset(self):
        self.obs = None
        self.motiv = None
        self.action = None
        self.full_action = None
        self.striatum.reset()
