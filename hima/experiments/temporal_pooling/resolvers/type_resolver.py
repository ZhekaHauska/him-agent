#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.config.types import TTypeOrFactory, LazyTypeResolver


class StpLazyTypeResolver(LazyTypeResolver):
    def __init__(self):
        super().__init__()
        self.resolvers_by_prefix = [
            ('block.', _resolve_block),
            ('encoder.', _resolve_encoder),
            ('stp.', _resolve_runner),
            ('ds.', _resolve_dataset),
            ('sp.', _resolve_spatial_pooler),
            ('tm.', _resolve_temporal_memory),
        ]

    def resolve(self, type_tag: str) -> TTypeOrFactory:
        assert type_tag is not None, 'Type is not specified'
        resolved_type = None
        for prefix, resolver in self.resolvers_by_prefix:
            if type_tag.startswith(prefix):
                resolved_type = resolver(type_tag)
                break

        if not resolved_type:
            raise ValueError(f'Unknown type tag: {type_tag}')
        return resolved_type


def _resolve_temporal_memory(type_tag: str):
    if type_tag == 'tm.general_feedback':
        from hima.experiments.temporal_pooling.blocks.general_feedback_tm import (
            GeneralFeedbackTemporalMemoryBlock
        )
        return GeneralFeedbackTemporalMemoryBlock


def _resolve_block(type_tag: str):
    if type_tag == 'block.storage':
        from hima.experiments.temporal_pooling.blocks.storage import StorageBlock
        return StorageBlock
    if type_tag == 'block.spatial_pooler':
        from hima.experiments.temporal_pooling.blocks.sp import SpatialPoolerBlock
        return SpatialPoolerBlock
    if type_tag == 'block.concatenator':
        from hima.experiments.temporal_pooling.blocks.concat import ConcatenatorBlock
        return ConcatenatorBlock
    if type_tag == 'block.sp':
        from hima.experiments.temporal_pooling.blocks.sp import SpatialPoolerBlock
        return SpatialPoolerBlock
    if type_tag == 'block.tm.general_feedback':
        from hima.experiments.temporal_pooling.blocks.general_feedback_tm import (
            GeneralFeedbackTemporalMemoryBlock
        )
        return GeneralFeedbackTemporalMemoryBlock


def _resolve_spatial_pooler(type_tag: str):
    if type_tag == 'sp.vectorized':
        from hima.experiments.temporal_pooling.stp.sp import SpatialPooler
        return SpatialPooler
    if type_tag == 'sp.list':
        from hima.experiments.temporal_pooling.stp.sp_list import SpatialPooler
        return SpatialPooler


def _resolve_dataset(type_tag):
    if type_tag == 'ds.synthetic_sequences':
        from hima.experiments.temporal_pooling.data.synthetic_sequences import SyntheticSequences
        return SyntheticSequences


def _resolve_encoder(type_tag):
    if type_tag == 'encoder.int_bucket':
        from hima.common.sdr_encoders import IntBucketEncoder
        return IntBucketEncoder
    if type_tag == 'encoder.int_random':
        from hima.common.sdr_encoders import IntRandomEncoder
        return IntRandomEncoder


def _resolve_runner(type_tag: str):
    if type_tag == 'stp.synthetic_sequences':
        from hima.experiments.temporal_pooling.test_stp import StpExperiment
        return StpExperiment

