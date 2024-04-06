#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from htm.bindings.sdr import SDR

from hima.common.sds import Sds
from hima.experiments.temporal_pooling.stp.sp import SpatialPooler
from hima.experiments.temporal_pooling.stp.sp_grouped import SpatialPoolerGrouped
from hima.experiments.temporal_pooling.stp.sp_grouped_float import (
    SpatialPoolerGrouped as FloatSpatialPoolerGrouped
)


class SpatialPoolerEnsemble:
    sps: list[SpatialPooler]

    def __init__(self, output_sds, seed, **kwargs):
        output_sds = Sds.make(output_sds)
        n_groups, single_sds = to_single_sds(output_sds)

        self.n_sp = n_groups
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.sps = [
            SpatialPooler(output_sds=single_sds, seed=self.rng.integers(1_000_000), **kwargs)
            for _ in range(self.n_sp)
        ]

        self.feedforward_sds = self.sps[0].feedforward_sds
        self.single_output_sds = self.sps[0].output_sds
        self.output_sds = output_sds

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

    @property
    def ff_avg_active_size(self):
        return self.sps[0].ff_avg_active_size

    def output_entropy(self):
        return np.mean([sp.output_entropy() for sp in self.sps])


class SpatialPoolerGroupedWrapper(SpatialPoolerGrouped):

    def __init__(self, seed, **kwargs):
        super().__init__(seed=seed, **kwargs)
        _, self.single_output_sds = to_single_sds(self.output_sds)

    def compute(self, input_sdr: SDR, learn: bool, output_sdr: SDR = None):
        if isinstance(input_sdr, SDR):
            input_sdr = input_sdr.sparse.copy()

        for _ in range(int(self.modulation)):
            result = super().compute(input_sdr, learn=learn)

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


class FloatSpatialPoolerGroupedWrapper(FloatSpatialPoolerGrouped):

    def __init__(self, seed, **kwargs):
        super().__init__(seed=seed, **kwargs)
        _, self.single_output_sds = to_single_sds(self.output_sds)

    def compute(self, input_sdr: SDR, learn: bool, output_sdr: SDR = None):
        if isinstance(input_sdr, SDR):
            input_sdr = input_sdr.sparse.copy()

        result = super().compute(input_sdr, learn=learn)

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


def to_single_sds(group_sds) -> tuple[int, Sds]:
    group_sds = Sds.make(group_sds)
    n_groups = group_sds.active_size

    single_shape = list(group_sds.shape)
    single_shape[-1] //= n_groups
    single_shape = tuple(single_shape)

    single_sds = Sds.make((single_shape, 1))

    return n_groups, single_sds


def to_group_sds(n_groups, single_sds) -> Sds:
    group_shape = list(single_sds.shape)
    group_shape[-1] *= n_groups
    group_shape = tuple(group_shape)

    group_sds = Sds.make((group_shape, n_groups))
    return group_sds
