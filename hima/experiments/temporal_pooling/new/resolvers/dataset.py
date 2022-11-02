#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config import extracted_type, resolve_init_params, TConfig
from hima.experiments.temporal_pooling.new.resolvers.graph import BlockResolver
from hima.experiments.temporal_pooling.new.blocks.graph import Block


class DataGeneratorResolver(BlockResolver):
    family = 'generator'

    @staticmethod
    def resolve(
            global_config: TConfig, config: TConfig, block_id: int, block_name: str, **kwargs
    ) -> Block:
        return DataGeneratorResolver._resolve(
            global_config=global_config, generator_config=config,
            block_id=block_id, block_name=block_name,
            **kwargs
        )

    @staticmethod
    def _resolve(
            global_config: TConfig, generator_config: TConfig,
            block_id: int, block_name: str, n_sequences: int,
            **induction_registry
    ):
        generator_config, generator_type = extracted_type(generator_config)

        if generator_type == 'synthetic_sequences':
            from hima.experiments.temporal_pooling.new.blocks.dataset_synth_sequences import (
                SyntheticSequencesGenerator
            )
            generator_config = resolve_init_params(generator_config, **induction_registry)
            generator = SyntheticSequencesGenerator(global_config, **generator_config)
        else:
            raise KeyError(f'{generator_type} is not supported')

        sequences = generator.generate_sequences(n_sequences)
        block = generator.make_block(block_id, block_name, sequences)
        return block
