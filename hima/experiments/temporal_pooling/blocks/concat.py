#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.sdr import SparseSdr
from hima.common.sdr_encoders import SdrConcatenator
from hima.experiments.temporal_pooling.blocks.graph import Block


class ConcatenatorBlock(Block):
    family = "concatenator"

    FEEDFORWARD = 'feedforward_#'
    OUTPUT = 'output'
    supported_streams = {FEEDFORWARD, OUTPUT}

    sdr_concatenator: SdrConcatenator

    def __init__(self, id: int, name: str):
        super(ConcatenatorBlock, self).__init__(id, name)

    def build(self):
        ff_sizes = [
            self.streams[stream].sds
            for stream in sorted(self.streams.keys())
            if stream.startswith(self._ff_pattern)
        ]
        self.sdr_concatenator = SdrConcatenator(ff_sizes)
        self.streams[self.OUTPUT].resolve_sds(self.sdr_concatenator.output_sds)

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        sdrs = [
            data[stream]
            for stream in sorted(data.keys())
            if stream.startswith(self._ff_pattern)
        ]
        self.streams[self.OUTPUT].sdr = self.sdr_concatenator.concatenate(*sdrs)

    @property
    def _ff_pattern(self) -> str:
        return self.FEEDFORWARD[:-1]
