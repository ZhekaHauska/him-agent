#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from typing import Union

from hima.common.config.values import get_unresolved_value
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.graph.node import Node
from hima.experiments.temporal_pooling.graph.stream import Stream


class Pipe(Node):
    """Pipe connects two blocks' streams. Thus, both streams operate in the same SDS."""

    src: Stream
    dst: Stream

    # TODO: implement delay and sdr bookkeeping
    delay: int
    _sdr: Union[SparseSdr, list[SparseSdr]]

    def __init__(self, src: Stream, dst: Stream, sds: Sds = get_unresolved_value()):
        self.src = src
        self.dst = dst

        self.src.join_sds(sds)
        self.src.align(self.dst)

    def expand(self):
        yield self

    def align_dimensions(self) -> bool:
        """
        Align dimensions of the streams connected via this pipe.
        Returns True if the streams' dimensions are resolved and correctly aligned,
        and False otherwise.
        """
        self.src.align(self.dst)
        return isinstance(self.src.sds, Sds)

    def forward(self) -> None:
        self.dst.sdr = self.src.sdr

    def __repr__(self) -> str:
        return f'{self.src} -> {self.dst} | {self.sds}'

    @property
    def sds(self):
        return self.src.sds
