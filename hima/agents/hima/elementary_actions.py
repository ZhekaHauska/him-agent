#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.sdr_encoders import IntBucketEncoder
import numpy as np


class ElementaryActions:
    def __init__(self, n_actions: int, bucket_size: int):
        self.n_actions = n_actions
        self.encoder = IntBucketEncoder(n_actions, bucket_size)
        # initialize patterns
        self.patterns = np.zeros((self.n_actions, self.encoder.output_sdr_size))
        for action in range(self.n_actions):
            pattern = self.encoder.encode(action)
            self.patterns[action, pattern] = 1

    def get_sparse_pattern(self, action):
        return self.encoder.encode(action)

    def get_dense_pattern(self, action):
        return self.patterns[action]
