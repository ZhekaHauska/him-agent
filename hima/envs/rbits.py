#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from sklearn.datasets import load_digits
import numpy as np

from hima.common.sds import Sds


class RandomBits:
    def __init__(self, sds: Sds, similarity_range: (float, float), seed=None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.sds = sds
        self.similarity_range = similarity_range
        self.reference_sdr = None
        self.similarity = None

        self.time_step = 0

    def obs(self):
        if self.reference_sdr is None:
            sdr = self._rng.choice(
                np.arange(self.sds.size),
                self.sds.active_size,
                replace=False
            )
        else:
            if self.similarity is None:
                similarity = self._rng.uniform(*self.similarity_range)
            else:
                similarity = self.similarity

            n_bits_to_preserve = int(round(similarity * self.sds.active_size))
            n_new_bits = self.sds.active_size - n_bits_to_preserve

            if n_bits_to_preserve > 0:
                old_bits = self._rng.choice(
                    self.reference_sdr,
                    n_bits_to_preserve,
                    replace=False
                )
            else:
                old_bits = np.empty(0)

            if n_new_bits > 0:
                new_bits = self._rng.choice(
                    np.arange(self.sds.size),
                    n_new_bits,
                    replace=False
                )
            else:
                new_bits = np.empty(0)

            sdr = np.concatenate([old_bits, new_bits])

        return sdr.astype('uint32')

    def act(self, reference_sdr, similarity):
        self.reference_sdr = reference_sdr
        self.similarity = similarity

    def step(self):
        self.time_step += 1

    def reset(self):
        self.time_step = 0
        self.reference_sdr = None
        self.similarity = None
