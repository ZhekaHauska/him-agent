#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.sdr_encoders import SdrConcatenator
from hima.experiments.temporal_pooling.graph.block import Block


class ConcatenatorBlock(Block):
    family = "concatenator"

    FEEDFORWARD = 'feedforward_#'
    OUTPUT = 'output'
    supported_streams = {FEEDFORWARD, OUTPUT}

    sdr_concatenator: SdrConcatenator

    def align_dimensions(self) -> bool:
        if not self.stream_registry[self.OUTPUT].valid and all(
            self.stream_registry[stream].valid
            for stream in self.stream_registry
            if stream.startswith(self._ff_pattern)
        ):
            ff_sizes = [
                self.stream_registry[stream].sds
                for stream in sorted(self.stream_registry.keys())
                if stream.startswith(self._ff_pattern)
            ]
            self.sdr_concatenator = SdrConcatenator(ff_sizes)
            self.stream_registry[self.OUTPUT].join_sds(self.sdr_concatenator.output_sds)

        return self.stream_registry[self.OUTPUT].valid

    def compile(self, **kwargs):
        pass

    def compute(self):
        sdrs = [
            self.stream_registry[stream].sdr
            for stream in sorted(self.stream_registry.keys())
            if stream.startswith(self._ff_pattern)
        ]
        self.stream_registry[self.OUTPUT].sdr = self.sdr_concatenator.concatenate(*sdrs)

    @property
    def _ff_pattern(self) -> str:
        return self.FEEDFORWARD[:-1]
