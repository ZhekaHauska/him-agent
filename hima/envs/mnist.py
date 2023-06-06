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
        self.images = self.digits.images
        self.target = self.digits.target
        self.size = self.target.size

        self.obs_shape = self.digits.images.shape[1:]
        self.order = np.arange(self.size)
        self.reset()
        self.time_step = 0
        self.cls = None

    def obs(self, return_class=False):
        idx = self.order[self.time_step % self.size]
        if return_class:
            return (
                self.images[idx],
                self.target[idx]
            )
        else:
            return self.images[idx]

    def act(self, cls: int):
        """
            Specify image class for environment to return.
            If None then any class will be returned.
        """
        self.cls = cls
        if cls is not None:
            mask = self.target == self.cls
            self.images = self.digits.images[mask]
            self.target = self.digits.target[mask]
        else:
            self.images = self.digits.images
            self.target = self.digits.target

        self.size = self.target.size

    def step(self):
        self.time_step += 1

    def reset(self):
        self.time_step = 0
        self._rng.shuffle(self.order)


if __name__ == '__main__':
    env = MNISTEnv()
