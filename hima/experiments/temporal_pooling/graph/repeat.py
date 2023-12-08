#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from textwrap import indent

from hima.experiments.temporal_pooling.graph.node import Node
from hima.experiments.temporal_pooling.graph.pipeline import Pipeline, ListIndentRest


class Repeat(Node):
    repeat: int
    do: Pipeline

    def __init__(self, repeat: int, do: Pipeline):
        self.repeat = repeat
        self.do = do

    def forward(self) -> None:
        for circle in range(self.repeat):
            self.do.forward()

    def __repr__(self) -> str:
        return '\n'.join([
            f'repeat: {self.repeat}',
            str(self.do)
        ])
