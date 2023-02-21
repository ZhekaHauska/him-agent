#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config.base import TConfig
from hima.common.config.values import resolve_init_params
from hima.experiments.temporal_pooling.resolvers.graph import BlockResolver
from hima.experiments.temporal_pooling.blocks.graph import Block


class DataGeneratorResolver(BlockResolver):
    family = 'generator'

    @staticmethod
    def resolve(
            global_config: TConfig, config: TConfig, block_id: int, block_name: str, **kwargs
    ) -> Block:
        return DataGeneratorResolver._resolve(
            global_config=global_config, generator_config=config,
            id=block_id, name=block_name,
            **kwargs
        )

    @staticmethod
    def _resolve(
            global_config: TConfig,
            id: int, name: str, n_sequences: int, **generator_config: TConfig
    ):
        from hima.experiments.temporal_pooling.blocks.dataset_synth_sequences import (
            SyntheticSequencesGenerator
        )
        generator = SyntheticSequencesGenerator(global_config, **generator_config)
        sequences = generator.generate_sequences(n_sequences)
        block = generator.make_block(id, name, sequences)
        return block
