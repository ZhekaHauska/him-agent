#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

import numpy as np
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR

from hima.common.config.base import TConfig
from hima.common.config.values import resolve_init_params
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.global_vars import VARS_LEARN

FEEDFORWARD = 'feedforward.sdr'
OUTPUT = 'output.sdr'


class SpatialPoolerBlock(Block):
    family = 'spatial_pooler.htm'
    supported_streams = {FEEDFORWARD, OUTPUT}

    sp: Any | TConfig
    _active_input: SDR
    _active_output: SDR

    def __init__(self, sp: TConfig, **kwargs):
        super().__init__(**kwargs)

        self.sp = self.model.config.config_resolver.resolve(sp, config_type=dict)

    def fit_dimensions(self) -> bool:
        return self[OUTPUT].valid_sds

    def compile(self):
        sp_config = self.sp
        feedforward_sds = self[FEEDFORWARD].sds
        output_sds = self[OUTPUT].sds

        sp_config = resolve_init_params(
            sp_config,
            inputDimensions=feedforward_sds.shape, potentialRadius=feedforward_sds.size,
            columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
        )
        self.sp = SpatialPooler(**sp_config)

        self._active_input = SDR(feedforward_sds.size)
        self._active_output = SDR(output_sds.size)

    def compute(self):
        learn = self[VARS_LEARN].get()
        self._active_input.sparse = self[FEEDFORWARD].get().copy()

        self.sp.compute(self._active_input, learn=learn, output=self._active_output)
        self[OUTPUT].set(np.array(self._active_output.sparse, copy=True))
