#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

from hima.common.config.base import extracted, TConfig
from hima.common.config.global_config import GlobalConfig
from hima.experiments.temporal_pooling.graph.block import Block


class SpatialPoolerBlock(Block):
    family = 'spatial_pooler'

    FEEDFORWARD = 'feedforward'
    OUTPUT = 'output'
    FEEDBACK = 'feedback'
    supported_streams = {FEEDFORWARD, OUTPUT, FEEDBACK}

    sp: Any

    def align_dimensions(self) -> bool:
        output = self.streams[self.OUTPUT]
        if output.valid and self.FEEDBACK in self.streams:
            self.streams[self.FEEDBACK].join_sds(output.sds)
        return output.valid

    def compile(self):
        self._compile(**self._config)

    def _compile(self, global_config: GlobalConfig, sp: TConfig):
        self.sp = global_config.resolve_object(
            sp,
            feedforward_sds=self.streams[self.FEEDFORWARD].sds,
            output_sds=self.streams[self.OUTPUT].sds
        )

    def compute(self, learn: bool = True):
        feedforward = self.streams[self.FEEDFORWARD].sdr
        self.streams[self.OUTPUT].sdr = self.sp.compute(feedforward, learn=learn)

    def switch_polarity(self):
        self.sp.polarity *= -1

    def compute_feedback(self):
        feedback = self.streams[self.FEEDBACK].sdr
        self.sp.process_feedback(feedback)
