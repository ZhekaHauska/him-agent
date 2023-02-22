#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.config.types import TTypeOrFactory, LazyTypeResolver


class StpLazyTypeResolver(LazyTypeResolver):
    def resolve(self, type_tag: str) -> TTypeOrFactory:
        if type_tag == 'stp.synthetic_sequences':
            from hima.experiments.temporal_pooling.test_stp import StpExperiment
            return StpExperiment
        if type_tag == 'storage':
            from hima.experiments.temporal_pooling.blocks.graph import StorageBlock
            return StorageBlock
        if type_tag == 'synthetic_sequences':
            from hima.experiments.temporal_pooling.blocks.dataset_synth_sequences import (
                SyntheticSequencesGenerator
            )
            return SyntheticSequencesGenerator
        if type_tag == 'int_bucket':
            from hima.common.sdr_encoders import IntBucketEncoder
            return IntBucketEncoder
        if type_tag == 'int_random':
            from hima.common.sdr_encoders import IntRandomEncoder
            return IntRandomEncoder
        if type_tag == 'custom_sp':
            from hima.experiments.temporal_pooling.blocks.custom_sp import CustomSpatialPoolerBlock
            return CustomSpatialPoolerBlock
        if type_tag == 'concatenator':
            from hima.experiments.temporal_pooling.blocks.concat import ConcatenatorBlock
            return ConcatenatorBlock

        raise ValueError(f'Unknown type tag: {type_tag}')
