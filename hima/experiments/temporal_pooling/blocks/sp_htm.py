#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR

from hima.common.config.values import resolve_init_params
from hima.common.config.base import extracted
from hima.common.sdr import SparseSdr
from hima.common.utils import timed
from hima.experiments.temporal_pooling.graph.block import Block


class SpatialPoolerBlock(Block):
    family = 'spatial_pooler.htm'

    FEEDFORWARD = 'feedforward'
    OUTPUT = 'output'
    supported_streams = {FEEDFORWARD, OUTPUT}

    sp: Any
    _active_input: SDR
    _active_output: SDR

    def __init__(self, id: int, name: str, **sp_config):
        super(SpatialPoolerBlock, self).__init__(id, name)

        sp_config, ff_sds, output_sds = extracted(sp_config, 'ff_sds', 'output_sds')

        self.register_stream(self.FEEDFORWARD).try_resolve_sds(ff_sds)
        self.register_stream(self.OUTPUT).try_resolve_sds(output_sds)

        self._sp_config = sp_config
        self.run_time = 0
        self.n_computes = 0

    def compile(self):
        sp_config = self._sp_config
        ff_sds = self.stream_registry[self.FEEDFORWARD].sds
        output_sds = self.stream_registry[self.OUTPUT].sds

        sp_config = resolve_init_params(
            sp_config,
            inputDimensions=ff_sds.shape, potentialRadius=ff_sds.size,
            columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
        )
        self.sp = SpatialPooler(**sp_config)

        self._active_input = SDR(ff_sds.size)
        self._active_output = SDR(output_sds.size)

    def compute(self):
        _, run_time = self._compute()
        self.run_time += run_time
        self.n_computes += 1

    @timed
    def _compute(self, learn: bool = True):
        self._active_input.sparse = feedforward.copy()
        self.sp.compute(self._active_input, learn=learn, output=self._active_output)
        self.stream_registry[self.OUTPUT].sdr = np.array(self._active_output.sparse, copy=True)
