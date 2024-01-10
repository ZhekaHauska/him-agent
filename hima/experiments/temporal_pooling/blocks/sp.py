#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

from hima.common.config.base import TConfig
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.global_vars import VARS_LEARN

FEEDFORWARD = 'feedforward.sdr'
OUTPUT = 'output.sdr'
FEEDBACK = 'feedback.sdr'


class SpatialPoolerBlock(Block):
    family = 'spatial_pooler'
    supported_streams = {FEEDFORWARD, OUTPUT, FEEDBACK}

    sp: Any | TConfig

    def __init__(self, sp: TConfig, **kwargs):
        super().__init__(**kwargs)
        self.sp = self.model.config.config_resolver.resolve(sp, config_type=dict)

    def fit_dimensions(self) -> bool:
        output, feedback = self[OUTPUT], self[FEEDBACK]
        if output.valid_sds and feedback is not None:
            feedback.set_sds(output.sds)
        return output.valid_sds

    def compile(self):
        self.sp = self.model.config.resolve_object(
            self.sp,
            feedforward_sds=self[FEEDFORWARD].sds,
            output_sds=self[OUTPUT].sds
        )

    def compute(self):
        learn = self.model.streams[VARS_LEARN].get()
        ff_sdr = self[FEEDFORWARD].get()
        output_sdr = self.sp.compute(ff_sdr, learn=learn)
        self[OUTPUT].set(output_sdr)

    def switch_polarity(self):
        self.sp.polarity *= -1

    def compute_feedback(self):
        fb_sdr = self[FEEDBACK].get()
        self.sp.process_feedback(fb_sdr)
