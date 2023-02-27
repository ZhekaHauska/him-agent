#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from textwrap import indent

from hima.experiments.temporal_pooling.graph.node import Node, ListIndentRest
from hima.experiments.temporal_pooling.graph.pipeline import Pipeline


class Repeat(Node):
    repeat: int
    pipeline: Pipeline

    def __init__(self, repeat: int, pipeline: Pipeline):
        self.repeat = repeat
        self.pipeline = pipeline

    def expand(self):
        return self.pipeline.expand()

    def align_dimensions(self) -> bool:
        return self.pipeline.align_dimensions()

    def forward(self) -> None:
        for circle in range(self.repeat):
            self.pipeline.forward()

    def __repr__(self) -> str:
        return '\n'.join([
            f'repeat: {self.repeat}',
            indent(str(self.pipeline), ListIndentRest)
        ])
