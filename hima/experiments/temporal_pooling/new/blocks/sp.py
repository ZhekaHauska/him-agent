#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR

from hima.common.config_utils import resolve_init_params, extracted
from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling.new.blocks.graph import Block


class SpatialPoolerBlock(Block):
    family = "spatial_pooler"

    FEEDFORWARD = 'feedforward'
    OUTPUT = 'output'
    supported_streams = {FEEDFORWARD, OUTPUT}

    sp: Any
    _active_input: SDR
    _active_output: SDR

    def __init__(self, id: int, name: str, **sp_config):
        super(SpatialPoolerBlock, self).__init__(id, name)

        sp_config, ff_sds, output_sds = extracted(sp_config, 'ff_sds', 'output_sds')

        self.register_stream(self.FEEDFORWARD).resolve_sds(ff_sds)
        self.register_stream(self.OUTPUT).resolve_sds(output_sds)

        self._sp_config = sp_config

    def build(self):
        sp_config = self._sp_config
        ff_sds = self.streams[self.FEEDFORWARD].sds
        output_sds = self.streams[self.OUTPUT].sds

        sp_config = resolve_init_params(
            sp_config,
            inputDimensions=ff_sds.shape, potentialRadius=ff_sds.size,
            columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
        )
        self.sp = SpatialPooler(**sp_config)

        self._active_input = SDR(ff_sds.size)
        self._active_output = SDR(output_sds.size)

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        self._compute(**data, **kwargs)

    def _compute(self, feedforward: SparseSdr, learn: bool = True):
        self._active_input.sparse = feedforward.copy()
        self.sp.compute(self._active_input, learn=learn, output=self._active_output)
        self.streams[self.OUTPUT].sdr = np.array(self._active_output.sparse, copy=True)
