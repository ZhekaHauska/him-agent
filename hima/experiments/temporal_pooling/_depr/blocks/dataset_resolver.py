#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.config import extracted_type, TConfig


def resolve_data_generator(config: TConfig, **induction_registry):
    generator_config, generator_type = extracted_type(config['generator'])

    if generator_type == 'aai_rotation':
        from hima.experiments.temporal_pooling._depr.blocks.dataset_aai import AAIRotationsGenerator
        return AAIRotationsGenerator(**generator_config)
    else:
        raise KeyError(f'{generator_type} is not supported')
