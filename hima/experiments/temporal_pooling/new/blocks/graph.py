#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from abc import ABC, abstractmethod
from typing import Union

from hima.common.config_utils import (
    resolve_value, is_resolved_value, get_unresolved_value
)
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import isnone


class Stream:
    """Stream defines the named dataflow to or from a block."""
    name: str
    sds: Sds
    sdr: SparseSdr
    block: 'Block'

    def __init__(self, name: str, block: 'Block'):
        self.block = block
        self.name = name
        self.sds = get_unresolved_value()
        self.sdr = []

    def resolve_sds(self, sds: Sds) -> Sds:
        self.sds = resolve_value(self.sds, substitute_with=sds)
        self.sds = Sds.as_sds(self.sds)
        if is_resolved_value(self.sds):
            self.block.on_stream_sds_resolved(self)
        return self.sds

    @property
    def fullname(self):
        return f'{self.block.name}.{self.name}'

    def __repr__(self):
        return self.fullname


class Block(ABC):
    """Base building block of the computational graph / neural network."""

    family: str = "base_block"
    supported_streams: set[str] = {}

    id: int
    name: str
    streams: dict[str, Stream]

    # TODO: log to charts, what to log?

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
        self.streams = {}

    def register_stream(self, name: str) -> 'Stream':
        if name not in self.streams:
            self.streams[name] = Stream(name=name, block=self)
        return self.streams[name]

    def on_stream_sds_resolved(self, stream: Stream):
        pass

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
    def tag(self):
        return f'{self.id}_{self.family}'

    def stream_tag(self, stream: str):
        return f'{self.tag}.{stream}'

    def __repr__(self):
        return f'{self.tag} {self.name}'


class ExternalApiBlock(Block):
    family = '_ext_api_'
    name = '___'

    def make_stream_stats_tracker(self, *, stream: str, stats_config, **kwargs):
        raise NotImplementedError()

    def build(self, **kwargs):
        pass

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        for stream_name in data:
            stream = self.streams[stream_name]
            stream.sdr = data[stream_name]


class Pipe:
    """Pipe connects two blocks' streams. Thus, both streams operate in the same SDS."""

    src: Stream
    dst: Stream

    # TODO: implement delay and sdr bookkeeping
    delay: int
    _sdr: Union[SparseSdr, list[SparseSdr]]

    def __init__(self, src: Stream, dst: Stream, sds: Sds = None):
        self.src = src
        self.dst = dst

        self.src.resolve_sds(isnone(sds, get_unresolved_value()))

    def forward(self):
        self.dst.sdr = self.src.sdr

    def align_dimensions(self) -> bool:
        if is_resolved_value(self.src.sds) and is_resolved_value(self.dst.sds):
            return True

        sds = self.src.sds
        sds = self.dst.resolve_sds(sds)
        sds = self.src.resolve_sds(sds)
        return is_resolved_value(sds)

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


class Pipeline:
    """
    Pipeline is the ordered traversal of the computational graph, that is it defines both —
    the graph itself and the order of the computations.
    """
    units: list[ComputationUnit]
    blocks: dict[str, Block]

    api: Block

    def __init__(self, units: list[ComputationUnit], blocks: dict[str, Block]):
        self.units = units
        self.blocks = blocks
        self.api = self.blocks[ExternalApiBlock.name]

    def step(self, input_data: dict[str, SparseSdr], **kwargs):
        # pass input data to the api block
        self.api.compute(input_data)

        for unit in self.units:
            unit.compute(**kwargs)

        return self.api.streams

    def __repr__(self):
        return f'{self.units}'
