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
            ('tracker.', _resolve_tracker),
            ('encoder.', _resolve_encoder),
            ('ds.', _resolve_dataset),
            ('sp.', _resolve_spatial_pooler),
            ('stp.', _resolve_spatial_temporal_pooler),
            ('tm.', _resolve_temporal_memory),
            ('stp_experiment.', _resolve_runner),
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


def _resolve_block(type_tag: str):
    if type_tag == 'block.storage':
        from hima.experiments.temporal_pooling.blocks.storage import StorageBlock
        return StorageBlock
    if type_tag == 'block.concatenator':
        from hima.experiments.temporal_pooling.blocks.concat import ConcatenatorBlock
        return ConcatenatorBlock
    if type_tag == 'block.sp':
        from hima.experiments.temporal_pooling.blocks.sp import SpatialPoolerBlock
        return SpatialPoolerBlock
    if type_tag == 'block.tm':
        from hima.experiments.temporal_pooling.blocks.tm import TemporalMemoryBlock
        return TemporalMemoryBlock
    if type_tag == 'block.lstm':
        from hima.experiments.temporal_pooling.blocks.lstm import LstmBlock
        return LstmBlock


def _resolve_temporal_memory(type_tag: str):
    if type_tag == 'tm.general_feedback':
        from hima.experiments.temporal_pooling.stp.general_feedback_tm import (
            GeneralFeedbackTM
        )
        return GeneralFeedbackTM
    if type_tag == 'tm.base':
        from hima.experiments.temporal_pooling.stp.temporal_memory import (
            TemporalMemory
        )
        return TemporalMemory
    if type_tag == 'tm.lstm':
        from hima.experiments.temporal_pooling.stp.lstm import LstmLayer
        return LstmLayer


def _resolve_spatial_pooler(type_tag: str):
    if type_tag == 'sp.vectorized':
        from hima.experiments.temporal_pooling.stp.sp import SpatialPooler
        return SpatialPooler
    if type_tag == 'sp.float':
        from hima.experiments.temporal_pooling.stp.sp_float import SpatialPooler
        return SpatialPooler
    if type_tag == 'sp.sdrr':
        from hima.experiments.temporal_pooling.stp.sp_rate import SpatialPooler
        return SpatialPooler
    if type_tag == 'sp.list':
        from hima.experiments.temporal_pooling.stp.sp_list import SpatialPooler
        return SpatialPooler
    if type_tag == 'sp.ensemble':
        from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerEnsemble
        return SpatialPoolerEnsemble
    if type_tag == 'sp.grouped':
        from hima.experiments.temporal_pooling.stp.sp_grouped import SpatialPoolerGrouped
        return SpatialPoolerGrouped


def _resolve_spatial_temporal_pooler(type_tag: str):
    if type_tag == 'stp.base':
        from hima.experiments.temporal_pooling.stp.stp import SpatialTemporalPooler
        return SpatialTemporalPooler


def _resolve_tracker(type_tag: str):
    if type_tag == 'tracker.sdr':
        from hima.experiments.temporal_pooling.stats.sdr_tracker import get_sdr_tracker
        return get_sdr_tracker
    if type_tag == 'tracker.tm':
        from hima.experiments.temporal_pooling.stats.tm_tracker import get_tm_tracker
        return get_tm_tracker


def _resolve_dataset(type_tag):
    if type_tag == 'ds.synthetic_sequences':
        from hima.experiments.temporal_pooling.data.synthetic_sequences import SyntheticSequences
        return SyntheticSequences
    if type_tag == 'ds.dvc_sequences':
        from hima.experiments.temporal_pooling.data.dvc import DvcSequences
        return DvcSequences
    if type_tag == 'ds.text_sequences':
        from hima.experiments.temporal_pooling.data.text import TextSequences
        return TextSequences


def _resolve_encoder(type_tag):
    if type_tag == 'encoder.int_bucket':
        from hima.common.sdr_encoders import IntBucketEncoder
        return IntBucketEncoder
    if type_tag == 'encoder.int_random':
        from hima.common.sdr_encoders import IntRandomEncoder
        return IntRandomEncoder


def _resolve_runner(type_tag: str):
    if type_tag == 'stp_experiment.synthetic_sequences':
        from hima.experiments.temporal_pooling.test_stp import StpExperiment
        return StpExperiment
    if type_tag == 'stp_experiment.sp_attractor.mnist':
        from hima.experiments.temporal_pooling.test_attractor_mnist import (
            SpAttractorMnistExperiment
        )
        return SpAttractorMnistExperiment
    if type_tag == 'stp_experiment.sp_attractor.rbits':
        from hima.experiments.temporal_pooling.test_attractor_rbits import (
            SpAttractorRandBitsExperiment
        )
        return SpAttractorRandBitsExperiment
    if type_tag == 'stp_experiment.sp_attractor':
        from hima.experiments.temporal_pooling.test_attractor import (
            SpAttractorExperiment
        )
        return SpAttractorExperiment
    if type_tag == 'stp_experiment.sp_encoder':
        from hima.experiments.temporal_pooling.test_encoder_mnist import (
            SpEncoderExperiment
        )
        return SpEncoderExperiment
    if type_tag == 'stp_experiment.sp_encoder_pinball':
        from hima.experiments.temporal_pooling.test_encoder_pinball import (
            SpEncoderPinballExperiment
        )
        return SpEncoderPinballExperiment

