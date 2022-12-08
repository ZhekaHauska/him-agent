#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config import resolve_init_params, TConfig
from hima.experiments.temporal_pooling.blocks.graph import Block
from hima.experiments.temporal_pooling.blocks.tp import TemporalPoolerBlock
from hima.experiments.temporal_pooling.resolvers.graph import BlockResolver


class TemporalPoolerResolver(BlockResolver):
    family = TemporalPoolerBlock.family

    @staticmethod
    def resolve(
            global_config: TConfig, config: TConfig, block_id: int, block_name: str, **kwargs
    ) -> Block:
        return TemporalPoolerResolver._resolve(
            tp_config=config,
            block_id=block_id, block_name=block_name,
            **kwargs
        )

    @staticmethod
    def _resolve(tp_config: TConfig, block_id: int, block_name: str, **induction_registry) -> Block:
        tp_config = resolve_init_params(tp_config, **induction_registry)
        return TemporalPoolerBlock(id=block_id, name=block_name, **tp_config)
