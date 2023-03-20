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
    do: Pipeline

    def __init__(self, repeat: int, do: Pipeline):
        self.repeat = repeat
        self.do = do

    def expand(self):
        return self.do.expand()

    def align_dimensions(self) -> bool:
        return self.do.align_dimensions()

    def forward(self) -> None:
        for circle in range(self.repeat):
            self.do.forward()

    def __repr__(self) -> str:
        return '\n'.join([
            f'repeat: {self.repeat}',
            indent(str(self.do), ListIndentRest)
        ])
