#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from textwrap import indent

from hima.experiments.temporal_pooling.graph.node import Node


ListIndentFirst = '  * '
ListIndentRest = '  '


class Pipeline(Node):
    """
    Pipeline is the named ordered list of computational graph nodes, which defines both —
    the graph itself and the order of the computations.
    """

    name: str
    _pipeline: list[Node]

    def __init__(self, name: str, pipeline: list):
        self.name = name
        self._pipeline = pipeline

    def forward(self) -> None:
        for node in self:
            node.forward()

    def __repr__(self):
        k = len(ListIndentRest)
        if len(self._pipeline) == 1:
            # for a single item list: do not print list
            node = self._pipeline[0]
            return '\n'.join([
                f'{self.name}:',
                indent(str(node), ListIndentRest)
            ])

        return '\n'.join([
            f'{self.name}:',
            '\n'.join([
                ListIndentFirst + indent(str(node), ListIndentRest)[k:]
                for node in self
            ])
        ])

    def __iter__(self):
        return iter(self._pipeline)
