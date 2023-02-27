#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from hima.common.config.utils import join_sds
from hima.common.config.values import get_unresolved_value
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds

# circular import otherwise
if TYPE_CHECKING:
    from hima.experiments.temporal_pooling.graph.block import Block


class Stream:
    """
    Stream defines the named dataflow to or from a block.

    While it can be compared to a port, it acts similar to a register — a memory slot for a data,
    i.e. a data is persisted and can be read several times until it's overwritten
    with the next value.
    """
    name: str
    sds: Sds
    sdr: SparseSdr
    block: Block

    def __init__(self, name: str, block: Block):
        assert block is not None, f'Stream {name} does not have block specified.'

        self.block = block
        self.name = name
        self.sds = get_unresolved_value()
        self.sdr = []

    @property
    def fullname(self):
        return f'{self.block.name}.{self.name}'

    def __repr__(self):
        return self.fullname

    def join_sds(self, sds: Sds | Any):
        # one-way apply
        self.sds = join_sds(self.sds, sds)

    def align(self, other: 'Stream'):
        # two-way exchange
        if self.valid and other.valid:
            assert self.sds == other.sds, f'Cannot align {self} and {other}.'
        elif self.valid:
            other.sds = self.sds
        elif other.valid:
            self.sds = other.sds

    @property
    def valid(self):
        return isinstance(self.sds, Sds)
