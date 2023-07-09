#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Callable

from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.node import Node


class BlockCall(Node):
    block: Block

    func: Callable
    func_name: str

    def __init__(self, block: Block, name: str):
        self.block = block
        self.func_name = name
        self.func = getattr(block, name)

    def forward(self) -> None:
        self.func()

    def __repr__(self) -> str:
        return f'{self.block.name}.{self.func_name}'
