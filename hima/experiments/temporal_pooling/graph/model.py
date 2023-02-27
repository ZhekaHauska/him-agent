#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.node import Node
from hima.experiments.temporal_pooling.graph.pipeline import Pipeline


class Model(Node):
    pipeline: Pipeline
    blocks: dict[str, Block]

    api: Block

    def __init__(
            self, api_block: str, pipeline: Pipeline, blocks: dict[str, Block]
    ):
        self.api = blocks[api_block]
        self.pipeline = pipeline
        self.blocks = blocks

    def expand(self):
        return self.pipeline.expand()

    def align_dimensions(self) -> bool:
        return self.pipeline.align_dimensions()

    def forward(self) -> None:
        self.pipeline.forward()

    def __repr__(self) -> str:
        return f'{self.pipeline}'
