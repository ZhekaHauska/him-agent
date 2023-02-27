#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config.values import is_resolved_value
from hima.common.config.base import extracted
from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.stream import Stream
from hima.experiments.temporal_pooling.stp.temporal_pooler import TemporalPooler


class TemporalPoolerBlock(Block):
    family = 'temporal_pooler'

    FEEDFORWARD = 'feedforward'
    OUTPUT = 'output'
    supported_streams = {FEEDFORWARD, OUTPUT}

    tp: TemporalPooler

    def __init__(self, id: int, name: str, **tp_config):
        super(TemporalPoolerBlock, self).__init__(id, name)

        tp_config, sds = extracted(tp_config, 'sds')

        self.register_stream(self.FEEDFORWARD).try_resolve_sds(sds)
        self.register_stream(self.OUTPUT).try_resolve_sds(sds)

        self._tp_config = tp_config

    def on_stream_sds_resolved(self, stream: Stream):
        # make sure that all streams have the same sds
        # DO NOT resolve if another stream is already resolved, cause finally their sds could
        #   be different (because of sparsity property of TP, defining output sparsity)
        propagate_to = self.FEEDFORWARD if stream.name == self.OUTPUT else self.OUTPUT
        if not is_resolved_value(self.streams[propagate_to].sds):
            self.streams[propagate_to].try_resolve_sds(stream.sds)

    def compile(self):
        sds = self.streams[self.FEEDFORWARD].sds
        self.tp = TemporalPooler(sds=sds, **self._tp_config)

        # correct output sds sparsity after TP initialization
        self.streams[self.OUTPUT].try_resolve_sds(self.tp.sds)

    def reset(self, **kwargs):
        self.tp.reset()
        super(TemporalPoolerBlock, self).reset(**kwargs)

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        self._compute(**data)

    def _compute(self, feedforward: SparseSdr, predicted_feedforward: SparseSdr = None):
        self.streams[self.OUTPUT].sdr = self.tp.compute(
            feedforward=feedforward,
            predicted_feedforward=predicted_feedforward,
        )
