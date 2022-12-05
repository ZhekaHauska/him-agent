#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR

from hima.common.config import resolve_init_params, extracted
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling._depr.blocks.base_block_stats import BlockStats
from hima.experiments.temporal_pooling.sdr_seq_stats import SdrSequenceStats


class SpatialPoolerBlockStats(BlockStats):
    seq_stats: SdrSequenceStats

    def __init__(self, output_sds: Sds):
        super(SpatialPoolerBlockStats, self).__init__(output_sds)
        self.seq_stats = SdrSequenceStats(self.output_sds)

    def update(self, current_output_sdr: SparseSdr):
        self.seq_stats.update(current_output_sdr)

    def step_metrics(self) -> dict[str, Any]:
        return self.seq_stats.step_metrics()

    def final_metrics(self) -> dict[str, Any]:
        return self.seq_stats.final_metrics()


class SpatialPoolerBlock:
    id: int
    name: str
    feedforward_sds: Sds
    output_sds: Sds

    output_sdr: SparseSdr
    sp: Any
    stats: SpatialPoolerBlockStats

    _active_input: SDR
    _active_output: SDR

    def __init__(self, feedforward_sds: Sds, output_sds: Sds, **sp_config):
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds
        self.sp = SpatialPooler(

            **sp_config
        )
        self.stats = SpatialPoolerBlockStats(self.output_sds)

        self.output_sdr = []
        self._active_input = SDR(self.feedforward_sds.size)
        self._active_output = SDR(self.output_sds.size)

    @property
    def tag(self) -> str:
        return f'{self.id}_sp'

    def reset(self):
        self._active_input.sparse = []
        self._active_output.sparse = []

    def reset_stats(self, stats: SpatialPoolerBlockStats = None):
        if stats is None:
            self.stats = SpatialPoolerBlockStats(self.output_sds)
        else:
            self.stats = stats

    def compute(self, active_input: SparseSdr, learn: bool = True) -> SparseSdr:
        self._active_input.sparse = active_input.copy()

        self.sp.compute(self._active_input, learn=learn, output=self._active_output)
        self.output_sdr = np.array(self._active_output.sparse, copy=True)

        self.stats.update(self.output_sdr)
        return self.output_sdr


def resolve_sp(sp_config, ff_sds: Sds, output_sds: Sds, seed: int):
    sp_config = resolve_init_params(
        sp_config, raise_if_not_resolved=False,
        ff_sds=ff_sds, output_sds=output_sds, seed=seed
    )
    sp_config, ff_sds, output_sds = extracted(sp_config, 'ff_sds', 'output_sds')

    # if FF/Out SDS was defined in config, they aren't Sds objects, hence explicit conversion
    ff_sds = Sds.as_sds(ff_sds)
    output_sds = Sds.as_sds(output_sds)

    sp_config = resolve_init_params(
        sp_config,
        inputDimensions=ff_sds.shape, potentialRadius=ff_sds.size,
        columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
        seed=seed
    )

    return SpatialPoolerBlock(feedforward_sds=ff_sds, output_sds=output_sds, **sp_config)
