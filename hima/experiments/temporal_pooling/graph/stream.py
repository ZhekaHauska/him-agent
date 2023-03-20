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


class Stream:
    """
    Stream defines the named dataflow to or from a block.

    While streams can be compared to ports, they act similar to registers — memory slots for a data,
    i.e. a data is persisted and can be read several times until it's overwritten
    with the next value.
    """
    registry: 'StreamRegistry'
    owner: Block | None
    name: str
    tracked: bool
    _value: Any

    def __init__(self, registry: 'StreamRegistry', name: str, block: Block = None):
        self.registry = registry
        self.owner = block
        self.name = name
        self.tracked = False
        self._value = None

    def get(self):
        return self._value

    def set(self, new_value, init=False):
        if init:
            self._value = new_value
        else:
            old_value, self._value = self._value, new_value
            if self.tracked:
                self.registry.on_stream_updated(self, old_value, new_value)

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

    def __init__(self, registry: 'StreamRegistry', name: str, block: Block = None):
        super().__init__(registry, name, block)
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


class StreamRegistry:
    _streams: dict[str, Stream | SdrStream]
    _trackers: dict[str, list]

    def __init__(self):
        self._streams = {}

    def __getitem__(self, item):
        return self._streams[item]

    def __setitem__(self, key, value):
        self._streams[key] = value

    def register(self, name: str, owner: Block = None) -> Stream | SdrStream:
        if name in self._streams:
            return self._streams[name]

        stream_class = SdrStream if name.endswith('.sdr') else Stream
        stream = stream_class(self, name, owner)
        self._streams[stream.name] = stream
        return stream

    def track(self, name: str, tracker):
        stream = self._streams.get(name, None)
        if not stream:
            stream = Stream(self, name)
            self._streams[name] = stream

        stream.tracked = True
        self._trackers.setdefault(name, []).append(tracker)

    def on_stream_updated(self, stream, old_value, new_value):
        for tracker in self._trackers[stream.name]:
            tracker.on_stream_updated(stream, old_value, new_value)

    def __iter__(self):
        yield from self._streams

    def __contains__(self, item):
        return item in self._streams

    def items(self):
        return self._streams.items()
