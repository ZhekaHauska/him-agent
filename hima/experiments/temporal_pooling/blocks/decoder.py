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
RAW_OUTPUT = 'raw_output'
FEEDBACK = 'feedback.sdr'


class DecoderBlock(Block):
    family = 'decoder'
    supported_streams = {FEEDFORWARD, OUTPUT, FEEDBACK, RAW_OUTPUT}

    decoder: Any | TConfig

    def __init__(self, decoder: TConfig, **kwargs):
        super().__init__(**kwargs)
        self.decoder = self.model.config.config_resolver.resolve(decoder, config_type=dict)
        self.register_stream(RAW_OUTPUT)

    def fit_dimensions(self) -> bool:
        output, feedback = self[OUTPUT], self[FEEDBACK]
        if output.valid_sds:
            feedback.set_sds(output.sds)
        if feedback.valid_sds:
            output.set_sds(feedback.sds)
        return output.valid_sds

    def compile(self):
        self.decoder = self.model.config.resolve_object(
            self.decoder,
            feedforward_sds=self[FEEDFORWARD].sds,
            output_sds=self[OUTPUT].sds
        )

    def decode(self):
        ff_sdr = self[FEEDFORWARD].get()
        output_dense = self.decoder.decode(ff_sdr)
        output_sdr = self.decoder.to_sdr(output_dense)

        self[RAW_OUTPUT].set(output_dense)
        self[OUTPUT].set(output_sdr)

    def learn(self):
        learn = self.model.streams[VARS_LEARN].get()
        if not learn:
            return

        ff_sdr = self[FEEDFORWARD].get()
        output_dense = self[RAW_OUTPUT].get()
        feedback_sdr = self[FEEDBACK].get()

        self.decoder.learn(ff_sdr, feedback_sdr, output_dense)
