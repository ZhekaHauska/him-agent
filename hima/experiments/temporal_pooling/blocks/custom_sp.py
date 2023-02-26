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
    sp_type: str

    def __init__(self, id: int, name: str, global_config=None, n_sequences=None, **sp_config):
        super(CustomSpatialPoolerBlock, self).__init__(id, name)

        sp_config, ff_sds, output_sds, self.sp_type = extracted(
            sp_config, 'ff_sds', 'output_sds', 'sp_type'
        )

        self.register_stream(self.FEEDFORWARD).try_resolve_sds(ff_sds)
        self.register_stream(self.OUTPUT).try_resolve_sds(output_sds)

        self._sp_config = sp_config

    def build(self):
        sp_config = self._sp_config
        ff_sds = self.streams[self.FEEDFORWARD].sds
        output_sds = self.streams[self.OUTPUT].sds

        sp_config = resolve_init_params(sp_config)
        if self.sp_type == 'vectorized':
            from hima.experiments.temporal_pooling.stp.custom_sp_vec import SpatialPooler
            self.sp = SpatialPooler(ff_sds=ff_sds, output_sds=output_sds, **sp_config)
        else:
            from hima.experiments.temporal_pooling.stp.custom_sp import SpatialPooler
            self.sp = SpatialPooler(ff_sds=ff_sds, output_sds=output_sds, **sp_config)

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        self._compute(**data, **kwargs)

    def _compute(self, feedforward: SparseSdr, learn: bool = True):
        self.streams[self.OUTPUT].sdr = self.sp.compute(feedforward, learn=learn)
