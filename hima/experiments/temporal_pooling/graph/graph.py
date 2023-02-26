#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.pipe import Pipe


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
