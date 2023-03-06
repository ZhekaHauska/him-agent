#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.experiments.temporal_pooling.graph.node import Node, ListIndentFirst


class Pipeline(Node):
    """
    Pipeline is the ordered traversal of the computational graph, that is it defines both —
    the graph itself and the order of the computations.
    """
    def __init__(self, pipeline):
        self._pipeline = list(pipeline)

    def expand(self):
        yield from self

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
        return '\n'.join([
            'pipeline:',
            '\n'.join(
                ListIndentFirst + str(node)
                for node in self
            )
        ])

    def __iter__(self):
        return iter(self._pipeline)
