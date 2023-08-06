#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from htm.bindings.sdr import SDR

from hima.common.sds import Sds
from hima.experiments.temporal_pooling.stp.sp import SpatialPooler


class SpatialPoolerEnsemble:
    sps: list[SpatialPooler]

    def __init__(self, n_sp, seed, **kwargs):
        self.n_sp = n_sp
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.sps = [
            SpatialPooler(**kwargs, seed=self.rng.integers(1_000_000))
            for _ in range(self.n_sp)
        ]

        self.feedforward_sds = self.sps[0].feedforward_sds
        self.single_output_sds = self.sps[0].output_sds

        shape = self.single_output_sds.shape
        ensemble_shape = (shape[0] * self.n_sp,) + shape[1:]
        self.output_sds = Sds.make((ensemble_shape, self.single_output_sds.sparsity))

    def compute(self, input_sdr: SDR, learn: bool, output_sdr: SDR = None):
        if isinstance(input_sdr, SDR):
            input_sdr = input_sdr.sparse.copy()

        index_shift = self.single_output_sds.size
        result = np.concatenate([
            sp.compute(input_sdr, learn=learn) + i * index_shift
            for i, sp in enumerate(self.sps)
        ])

        if output_sdr is not None:
            output_sdr.sparse = result
        return result

    def getSingleNumColumns(self):
        return self.single_output_sds.size

    def getSingleColumnsDimensions(self):
        return self.single_output_sds.shape

    def getColumnsDimensions(self):
        return self.output_sds.shape

    def getNumColumns(self):
        return self.output_sds.size

    def getInputDimensions(self):
        return self.feedforward_sds.shape

    def getNumInputs(self):
        return self.feedforward_sds.size

    def output_entropy(self):
        return np.mean([sp.output_entropy() for sp in self.sps])
