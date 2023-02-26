#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Any

from hima.common.config.utils import join_sds
from hima.common.config.values import get_unresolved_value
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


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
    block: 'Block'

    def __init__(self, name: str, block: 'Block'):
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
        x_is_sds = isinstance(self.sds, Sds)
        y_is_sds = isinstance(other.sds, Sds)

        if x_is_sds and y_is_sds:
            assert self.sds == other.sds, f'Cannot align {self} and {other}.'
        elif x_is_sds:
            other.sds = self.sds
        elif y_is_sds:
            self.sds = other.sds


class Block(ABC):
    """Base building block of the computational graph / neural network."""

    family: str = "base_block"
    supported_streams: set[str] = {}

    id: int
    name: str
    streams: dict[str, Stream]

    # TODO:
    #  1. log to charts, what to log?
    #  2. rename tag to ? and consider removing id

    def __init__(self, id: int, name: str, **kwargs):
        self.id = id
        self.name = name
        self.streams = {}
        self._parse_streams(kwargs)

    def register_stream(self, name: str) -> Stream:
        if name not in self.streams:
            self.streams[name] = Stream(name=name, block=self)
        return self.streams[name]

    # --------------- Overrideable public interface ---------------

    # noinspection PyMethodMayBeStatic
    def align_dimensions(self) -> bool:
        """
        Align or induce block's streams dimensions.
        By default, does nothing. Override if it's applicable.
        """
        return True

    def reset(self, **kwargs):
        for name in self.streams:
            self.streams[name].sdr = []

    @abstractmethod
    def build(self, **kwargs):
        """Build block after all its configurable parameters are resolved."""
        raise NotImplementedError()

    @abstractmethod
    def compute(self, data: dict[str, SparseSdr], **kwargs):
        """Make a computation given the provided input data streams."""
        raise NotImplementedError()

    # --------------- String representation ---------------

    @property
    def shortname(self):
        return f'{self.id}_{self.family}'

    @property
    def fullname(self):
        return f'{self.shortname} {self.name}'

    def __repr__(self):
        return self.fullname

    def _parse_streams(self, kwargs: dict):
        for key, value in kwargs.items():
            if not str.endswith(key, '_sds'):
                continue

            stream_name, sds = key[:-4], value
            stream = self.register_stream(stream_name)
            stream.join_sds(sds)


class Pipe:
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

    def forward(self):
        self.dst.sdr = self.src.sdr

    def align_dimensions(self) -> bool:
        """
        Align dimensions of the streams connected via this pipe.
        Returns True if the streams' dimensions are resolved and correctly aligned,
        and False otherwise.
        """
        self.src.align(self.dst)
        return isinstance(self.src.sds, Sds)

    @property
    def sds(self):
        return self.src.sds

    def __repr__(self):
        return f'{self.sds}| {self.src} -> {self.dst}'


class ComputationUnit:
    """
    Computational unit defines a node in the computational graph.
    It joins several input connections of a block to represent a single computational unit —
    connections provide the input data for a single computation inside the block.
    """

    block: Block
    connections: list[Pipe]

    def __init__(self, connections: list[Pipe]):
        self.connections = connections
        self.block = connections[0].dst.block

    def compute(self, **kwargs):
        for connection in self.connections:
            connection.forward()

        input_data = {
            connection.dst.name: connection.dst.sdr
            for connection in self.connections
        }
        return self.block.compute(input_data, **kwargs)

    def __repr__(self):
        if len(self.connections) == 1:
            return f'{self.connections[0]}'
        else:
            return f'{self.connections}'


class Model:
    """
    Pipeline is the ordered traversal of the computational graph, that is it defines both —
    the graph itself and the order of the computations.
    """

    pipeline: list[ComputationUnit]
    blocks: dict[str, Block]

    api: Block

    def __init__(self, api_block: str, units: list[ComputationUnit], blocks: dict[str, Block]):
        self.pipeline = units
        self.blocks = blocks
        self.api = [block for block in blocks.values() if block.in_out][0]

    def step(self, input_data: dict[str, SparseSdr], **kwargs):
        # pass input data to the api block
        self.api.compute(input_data)

        for unit in self.pipeline:
            unit.compute(**kwargs)

        return self.api.streams

    def __repr__(self):
        return f'{self.pipeline}'
