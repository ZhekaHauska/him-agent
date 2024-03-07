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
    from hima.experiments.temporal_pooling.blocks.tracker import TGeneralHandler


class Stream:
    """
    Stream defines a named variable.

    While streams can be compared to ports, they also act similar to memory registers —
    memory slots for a data, i.e. a data is persisted and can be read several
    times until it's overwritten with the next value.

    A stream may have an owner (a block), in which case it is a part of the block's
    public API I/O, i.e. dataflow to/from this block.

    All streams in a model are public/global, i.e. they can be referenced from anywhere
    by their full name. However, by convention, each block should reference its own streams only,
    while communication between different blocks should be defined in the config. This convention
    restricts the referencing by the full name only for the config, and everywhere else (in the
    code) streams are referenced by their short name.

    Otherwise, we would have to hardcode the block names. We should avoid it anytime!

    Streams without an owner block are a bit special, because their name do not bind with
    an owner's name. Which means you can freely reference them from anywhere.
    There are two kinds of such streams: commonly used and auxiliary.

    I call commonly used streams without an owner as _global variables_. They are defined in
    `global_vars.py`. Still, it's better to avoid both: hardcode their string names
    (use corresponding vars) and require their presence (require only when it's necessary).

    Auxiliary streams are mostly used in a model's graph just like regular local variables
    in python to store some input/output or intermediate results (for stats tracking purposes
    included). Because of that, I call them `local variables`. It is expected that you define
    and use them only in the model graph in the config. That is, avoid referencing them
    in the code, at least outside the experiment runner's code.
    """
    owner: Block | None
    name: str
    _value: Any
    _trackers: list[TGeneralHandler]

    def __init__(self, name: str, block: Block = None):
        self.owner = block
        self.name = name
        self._trackers = []
        self._value = None

    # ====== STREAM I/O IS EXPECTED VIA THE FOLLOWING GET/SET METHODS ONLY =========
    def get(self):
        return self._value

    def set(self, value=None, reset=False):
        self._value = value

        # notify all stream trackers
        for tracker in self._trackers:
            tracker(self, value, reset)
    # ==============================================================================

    def track(self, tracker: TGeneralHandler):
        """Set the stream tracker. It will be notified any time the stream is changed."""
        self._trackers.append(tracker)

    @property
    def is_sdr(self):
        return False

    def __repr__(self):
        return self.name


class SdrStream(Stream):
    """A special case of a stream that operates with SDR values."""
    sds: Sds

    def __init__(self, name: str, block: Block = None):
        super().__init__(name, block)
        self.sds = get_unresolved_value()
        self._value = []

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
