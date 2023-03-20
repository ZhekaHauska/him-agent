#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from textwrap import indent

from hima.experiments.temporal_pooling.graph.node import Node, ListIndentFirst, ListIndentRest


class Pipeline(Node):
    """
    Pipeline is the ordered traversal of the computational graph, that is it defines both —
    the graph itself and the order of the computations.
    """

    name: str
    _pipeline: list[Node]

    def __init__(self, name: str, pipeline: list):
        self.name = name
        self._pipeline = pipeline

    def expand(self):
        yield from self
        # for node in self:
        #     yield from node.expand()

    def align_dimensions(self) -> bool:
        # cannot reduce to all(...) as it shortcuts on the first returned False
        # hence, doesn't call align for the rest of the pipeline
        aligned = True
        for node in self:
            aligned &= node.align_dimensions()
        return aligned

    def forward(self) -> None:
        for node in self:
            node.forward()

    def __repr__(self):
        k = len(ListIndentFirst)
        return '\n'.join([
            f'{self.name}:',
            '\n'.join([
                ListIndentFirst + indent(str(node), ListIndentRest)[k:]
                for node in self
            ])
        ])

    def __iter__(self):
        return iter(self._pipeline)

    @staticmethod
    def extract_args(**kwargs) -> tuple[str, list]:
        assert len(kwargs) == 1
        for pipeline_name, pipeline in kwargs.items():
            return pipeline_name, list(pipeline)
