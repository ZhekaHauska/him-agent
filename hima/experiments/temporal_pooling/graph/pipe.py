#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from hima.common.config.values import get_unresolved_value
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.graph.node import Node
from hima.experiments.temporal_pooling.graph.stream import Stream, SdrStream


class Pipe(Node):
    """Pipe connects two blocks' streams. Thus, both streams operate in the same SDS."""

    src: Stream | SdrStream
    dst: Stream | SdrStream

    # TODO: implement delay and bookkeeping
    delay: int

    def __init__(self, src: Stream, dst: Stream, sds: Sds = get_unresolved_value()):
        assert src.is_sdr == dst.is_sdr

        self.src = src
        self.dst = dst

        if self.src.is_sdr:
            self.src.set_sds(sds)
            self.src.exchange_sds(self.dst)

    def expand(self):
        yield self

    def align_dimensions(self) -> bool:
        """
        Align dimensions of the streams connected via this pipe.
        Returns True if the streams' dimensions are resolved and correctly aligned,
        and False otherwise.
        """
        if self.src.is_sdr:
            self.src: SdrStream
            self.src.exchange_sds(self.dst)
            return self.src.valid_sds
        return True

    def forward(self) -> None:
        self.dst.set(self.src.get())

    def __repr__(self) -> str:
        res = f'{self.src} -> {self.dst}'
        if self.src.is_sdr:
            res = f'{res} | {self.src.sds}'
        return res
