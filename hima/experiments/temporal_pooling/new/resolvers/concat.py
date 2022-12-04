#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config import TConfig
from hima.experiments.temporal_pooling.new.blocks.concat import ConcatenatorBlock
from hima.experiments.temporal_pooling.new.blocks.graph import Block
from hima.experiments.temporal_pooling.new.resolvers.graph import BlockResolver


class ConcatenatorResolver(BlockResolver):
    family = ConcatenatorBlock.family

    @staticmethod
    def resolve(
            global_config: TConfig, config: TConfig, block_id: int, block_name: str, **kwargs
    ) -> Block:
        return ConcatenatorBlock(id=block_id, name=block_name)
