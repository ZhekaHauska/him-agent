#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from typing import Literal


class DVS:
    def __init__(self, shape, mode: Literal['abs', 'clip'] = 'abs', seed=None):
        self.shape = shape
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.mode = mode

        self.initial_previous_image = self._rng.random(self.shape)
        self.prev_image = self.initial_previous_image

    def capture(self, data):
        gray = np.dot(data[:, :, :3], [299 / 1000, 587 / 1000, 114 / 1000])

        if self.mode == 'abs':
            diff = np.abs(gray - self.prev_image)
        elif self.mode == 'clip':
            diff = np.clip(gray - self.prev_image, 0, None)
        else:
            raise ValueError(f'There is no such mode: "{self.mode}"!')

        self.prev_image = gray.copy()

        thresh = diff.mean()
        events = np.flatnonzero(diff > thresh)

        return events

    def reset(self):
        self.prev_image = self.initial_previous_image.copy()
