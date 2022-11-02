#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config import resolve_init_params, TConfig
from hima.experiments.temporal_pooling.new.blocks.graph import Block
from hima.experiments.temporal_pooling.new.blocks.stp import SpatiotemporalPoolerBlock
from hima.experiments.temporal_pooling.new.resolvers.graph import BlockResolver


class SpatiotemporalPoolerResolver(BlockResolver):
    family = SpatiotemporalPoolerBlock.family

    @staticmethod
    def resolve(
            global_config: TConfig, config: TConfig, block_id: int, block_name: str, **kwargs
    ) -> Block:
        return SpatiotemporalPoolerResolver._resolve(
            stp_config=config,
            block_id=block_id, block_name=block_name,
            **kwargs
        )

    @staticmethod
    def _resolve(
            stp_config: TConfig, block_id: int, block_name: str, **induction_registry
    ) -> Block:
        stp_config = resolve_init_params(stp_config, **induction_registry)
        return SpatiotemporalPoolerBlock(id=block_id, name=block_name, **stp_config)
