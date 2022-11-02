#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.config import extracted_type, resolve_init_params


def resolve_encoder(
        config: dict, key,
        n_values: int = None, active_size: int = None, seed: int = None
):
    encoder_registry = config['encoder']
    encoder_config, encoder_type = extracted_type(encoder_registry[key])

    if encoder_type == 'int_bucket':
        from hima.common.sdr_encoders import IntBucketEncoder
        encoder_config = resolve_init_params(
            encoder_config, n_values=n_values, bucket_size=active_size
        )
        return IntBucketEncoder(**encoder_config)
    if encoder_type == 'int_random':
        from hima.common.sdr_encoders import IntRandomEncoder
        encoder_config = resolve_init_params(
            encoder_config, n_values=n_values, active_size=active_size, seed=seed
        )
        return IntRandomEncoder(**encoder_config)
    else:
        raise KeyError(f'{encoder_type} is not supported')
