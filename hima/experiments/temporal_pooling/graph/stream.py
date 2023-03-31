#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hima.common.config.utils import join_sds
from hima.common.config.values import get_unresolved_value
from hima.common.sds import Sds

# circular import otherwise
if TYPE_CHECKING:
    from hima.experiments.temporal_pooling.graph.block import Block
    from hima.experiments.temporal_pooling.stats.alt_tracker import THandler


class Stream:
    """
    Stream defines the named dataflow to or from a block.

    While streams can be compared to ports, they act similar to registers — memory slots for a data,
    i.e. a data is persisted and can be read several times until it's overwritten
    with the next value.
    """
    owner: Block | None
    name: str
    _value: Any
    _trackers: list[THandler]

    def __init__(self, name: str, block: Block = None):
        self.owner = block
        self.name = name
        self._trackers = []
        self._value = None

    def get(self):
        return self._value

    def set(self, value, reset=False):
        self._value = value

        for tracker in self._trackers:
            tracker(self, value, reset)

    def track(self, tracker: THandler):
        self._trackers.append(tracker)

    @property
    def is_sdr(self):
        return False

    @property
    def fullname(self):
        return self.name

    def __repr__(self):
        return self.fullname


class SdrStream(Stream):
    sds: Sds

    def __init__(self, name: str, block: Block = None):
        super().__init__(name, block)
        self.sds = get_unresolved_value()

    @property
    def is_sdr(self):
        return True

    def set_sds(self, sds: Sds | Any):
        # one-way apply
        self.sds = join_sds(self.sds, sds)

    def exchange_sds(self, other: 'SdrStream' | Stream):
        assert other.is_sdr

        # two-way exchange
        if self.valid_sds and other.valid_sds:
            assert self.sds == other.sds, f'Cannot align {self} and {other}.'
        elif self.valid_sds:
            other.sds = self.sds
        elif other.valid_sds:
            self.sds = other.sds

    @property
    def valid_sds(self):
        return isinstance(self.sds, Sds)
