#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from htm.bindings.sdr import SDR

from hima.experiments.temporal_pooling.stp.sp import SpatialPooler


class SpatialPoolerEnsemble:
    sps: list[SpatialPooler]

    def __init__(self, n_sp, **kwargs):
        self.n_sp = n_sp
        self.seed = kwargs.pop('seed')
        self._rng = np.random.default_rng(self.seed)

        self.sps = [
            SpatialPooler(**kwargs, seed=self._rng.integers(1_000_000))
            for _ in range(self.n_sp)
        ]

    def compute(self, input_sdr: SDR, learn: bool, output_sdr: SDR):
        input_sdr = input_sdr.sparse.copy()
        index_shift = self.getSingleNumColumns()

        output_sdr.sparse = np.concatenate([
            sp.compute(input_sdr, learn=learn) + i * index_shift
            for i, sp in enumerate(self.sps)
        ])

    def getSingleNumColumns(self):
        return self.sps[0].output_size

    def getSingleColumnsDimensions(self):
        return self.sps[0].output_sds.shape

    def getNumColumns(self):
        return self.getSingleNumColumns() * self.n_sp

    def getNumInputs(self):
        return self.sps[0].ff_size

    def getColumnDimensions(self):
        return [sp.output_sds.shape for sp in self.sps]

    def getInputDimensions(self):
        return self.sps[0].feedforward_sds.shape
