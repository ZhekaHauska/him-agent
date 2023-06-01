#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from sklearn.datasets import load_digits
import numpy as np


class MNISTEnv:
    def __init__(self, seed=None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.digits = load_digits()
        self.size = self.digits.images.shape[0]
        self.obs_shape = self.digits.images.shape[1:]
        self.order = np.arange(self.size)
        self.reset()
        self.time_step = 0

    def obs(self):
        return self.digits.images[self.time_step % self.size]

    def act(self):
        pass

    def step(self):
        self.time_step += 1

    def reset(self):
        self.time_step = 0
        self._rng.shuffle(self.order)


if __name__ == '__main__':
    env = MNISTEnv()
