#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config.base import TConfig
from hima.common.config.values import resolve_init_params
from hima.experiments.temporal_pooling.blocks.custom_sp import CustomSpatialPoolerBlock
from hima.experiments.temporal_pooling.graph.graph import Block
from hima.experiments.temporal_pooling.resolvers.graph import BlockResolver


class CustomSpatialPoolerResolver(BlockResolver):
    family = CustomSpatialPoolerBlock.family

    @staticmethod
    def resolve(
            global_config: TConfig, config: TConfig, block_id: int, block_name: str, **kwargs
    ) -> Block:
        return CustomSpatialPoolerResolver._resolve(
            sp_config=config,
            block_id=block_id, block_name=block_name,
            **kwargs
        )

    @staticmethod
    def _resolve(sp_config: TConfig, block_id: int, block_name: str, **induction_registry) -> Block:
        sp_config = resolve_init_params(sp_config, **induction_registry)
        return CustomSpatialPoolerBlock(id=block_id, name=block_name, **sp_config)
