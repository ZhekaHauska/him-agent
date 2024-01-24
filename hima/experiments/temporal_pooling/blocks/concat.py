#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from hima.common.sdr_encoders import SdrConcatenator
from hima.experiments.temporal_pooling.graph.block import Block

FEEDFORWARD = 'feedforward_#*.sdr'
OUTPUT = 'output.sdr'


class ConcatenatorBlock(Block):
    family = "concatenator"

    supported_streams = {OUTPUT}

    sdr_concatenator: SdrConcatenator | None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sdr_concatenator = None
        self.ff_streams = None

    def fit_dimensions(self) -> bool:
        if self.ff_streams is None:
            ff_pattern = self.to_full_stream_name(self._ff_pattern)
            prefix_len = len(self.name) + 1

            self.ff_streams = sorted(
                stream_name[prefix_len:]
                for stream_name in self.model.streams
                if stream_name.startswith(ff_pattern)
            )
            # IMPORTANT: add dynamically induced feedforward streams to supported streams
            for stream_name in self.ff_streams:
                self.supported_streams[stream_name] = self.to_full_stream_name(stream_name)

        if not self[OUTPUT].valid_sds and all(
            self[stream_name].valid_sds
            for stream_name in self.ff_streams
        ):
            ff_sizes = [
                self[stream_name].sds
                for stream_name in self.ff_streams
            ]
            self.sdr_concatenator = SdrConcatenator(ff_sizes)
            self[OUTPUT].set_sds(self.sdr_concatenator.output_sds)

        return self[OUTPUT].valid_sds

    def compile(self, **kwargs):
        pass

    def compute(self):
        sdrs = [
            self[stream_name].get()
            for stream_name in self.ff_streams
        ]
        self[OUTPUT].set(self.sdr_concatenator.concatenate(*sdrs))

    @property
    def _ff_pattern(self) -> str:
        return FEEDFORWARD[:-5]
