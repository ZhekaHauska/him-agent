#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config_utils import TConfig, resolve_init_params
from hima.experiments.temporal_pooling.new.blocks.sp import SpatialPoolerBlock
from hima.experiments.temporal_pooling.new.blocks.graph import Block
from hima.experiments.temporal_pooling.new.resolvers.graph import BlockResolver


class SpatialPoolerResolver(BlockResolver):
    family = SpatialPoolerBlock.family

    @staticmethod
    def resolve(
            global_config: TConfig, config: TConfig, block_id: int, block_name: str, **kwargs
    ) -> Block:
        return SpatialPoolerResolver._resolve(
            sp_config=config,
            block_id=block_id, block_name=block_name,
            **kwargs
        )

    @staticmethod
    def _resolve(sp_config: TConfig, block_id: int, block_name: str, **induction_registry) -> Block:
        sp_config = resolve_init_params(sp_config, **induction_registry)
        return SpatialPoolerBlock(id=block_id, name=block_name, **sp_config)
