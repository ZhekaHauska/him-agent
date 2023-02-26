#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling.graph.block import Block


class StorageBlock(Block):
    family = "storage"

    def build(self, **kwargs):
        pass

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        # put data to the specified streams
        for stream_name in data:
            self.streams[stream_name].sdr = data[stream_name]
