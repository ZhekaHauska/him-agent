#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

from hima.common.config.values import resolve_init_params
from hima.common.config.base import extracted
from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling.graph.block import Block


class CustomSpatialPoolerBlock(Block):
    family = "custom_sp"

    FEEDFORWARD = 'feedforward'
    OUTPUT = 'output'
    supported_streams = {FEEDFORWARD, OUTPUT}

    sp: Any

    def compile(self):
        sp_config = self._config
        feedforward_sds = self.streams[self.FEEDFORWARD].sds
        output_sds = self.streams[self.OUTPUT].sds

        sp_config, sp_type = extracted(sp_config, 'sp_type')
        if sp_type == 'vectorized':
            from hima.experiments.temporal_pooling.stp.custom_sp_vec import SpatialPooler
            self.sp = SpatialPooler(
                feedforward_sds=feedforward_sds, output_sds=output_sds, **sp_config
            )
        else:
            from hima.experiments.temporal_pooling.stp.custom_sp import SpatialPooler
            self.sp = SpatialPooler(
                feedforward_sds=feedforward_sds, output_sds=output_sds, **sp_config
            )

    def compute(self, learn: bool = True):
        feedforward = self.streams[self.FEEDFORWARD].sdr
        self.streams[self.OUTPUT].sdr = self.sp.compute(feedforward, learn=learn)
