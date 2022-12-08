#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.blocks.graph import Stream
from hima.experiments.temporal_pooling.stats.config import StatsMetricsConfig
from hima.experiments.temporal_pooling.stats.tracker import Tracker, TMetrics


class StreamTracker(Tracker):
    stream: Stream
    trackers: list[Tracker]

    # TODO: history of trackers?

    def __init__(
            self, stream: Stream, trackers: list[str], config: StatsMetricsConfig,
            n_sequences: int
    ):
        self.stream = stream
        print(stream, trackers)
        self.trackers = [
            resolve_tracker(
                tracker_name=tracker_name, sds=stream.sds, n_sequences=n_sequences,
                stats_config=config
            )
            for tracker_name in trackers
        ]

    @property
    def name(self):
        return self.stream.fullname

    @property
    def sds(self):
        return self.stream.sds

    def on_sequence_started(self, sequence_id: int):
        for tracker in self.trackers:
            tracker.on_sequence_started(sequence_id)

    def on_epoch_started(self):
        for tracker in self.trackers:
            tracker.on_epoch_started()

    def on_step(self, sdr: SparseSdr):
        for tracker in self.trackers:
            tracker.on_step(sdr)

    def on_sequence_finished(self):
        for tracker in self.trackers:
            tracker.on_sequence_finished()

    def on_epoch_finished(self):
        for tracker in self.trackers:
            tracker.on_epoch_finished()

    def step_metrics(self) -> TMetrics:
        result = {}
        for tracker in self.trackers:
            result.update(tracker.step_metrics())
        result = rename_dict_keys(result, add_prefix=f'{self.name}/')
        return result

    def aggregate_metrics(self) -> TMetrics:
        result = {}
        for tracker in self.trackers:
            result.update(tracker.aggregate_metrics())
        result = rename_dict_keys(result, add_prefix=f'{self.name}/')
        return result


def resolve_tracker(
        tracker_name: str, sds: Sds, n_sequences: int, stats_config: StatsMetricsConfig
) -> Tracker:
    if tracker_name == 'sdr':
        from hima.experiments.temporal_pooling.stats.sdr_tracker import SdrTracker
        return SdrTracker(sds)
    elif tracker_name == 'cross.offline.el':
        from hima.experiments.temporal_pooling.stats.similarity_matrix import \
            OfflineElementwiseSimilarityMatrix
        return OfflineElementwiseSimilarityMatrix(
            n_sequences=n_sequences, sds=sds,
            normalization=stats_config.mae_normalization,
            symmetrical=stats_config.symmetrical_similarity
        )
    elif tracker_name == 'cross.offline.prefix':
        from hima.experiments.temporal_pooling.stats.similarity_matrix import \
            OfflinePmfSimilarityMatrix
        return OfflinePmfSimilarityMatrix(
            n_sequences=n_sequences, sds=sds,
            normalization=stats_config.mae_normalization,
            symmetrical=stats_config.symmetrical_similarity,
            distribution_metrics=stats_config.distribution_metrics,
            pmf_decay=stats_config.pmf_decay
        )
    elif tracker_name == 'cross.online.el':
        from hima.experiments.temporal_pooling.stats.similarity_matrix import \
            OnlineElementwiseSimilarityMatrix
        return OnlineElementwiseSimilarityMatrix(
            n_sequences=n_sequences, sds=sds,
            normalization=stats_config.mae_normalization,
            symmetrical=stats_config.symmetrical_similarity,
            online_similarity_decay=stats_config.online_similarity_decay
        )
    elif tracker_name == 'cross.online.prefix':
        from hima.experiments.temporal_pooling.stats.similarity_matrix import \
            OnlinePmfSimilarityMatrix
        return OnlinePmfSimilarityMatrix(
            n_sequences=n_sequences, sds=sds,
            normalization=stats_config.mae_normalization,
            symmetrical=stats_config.symmetrical_similarity,
            distribution_metrics=stats_config.distribution_metrics,
            online_similarity_decay=stats_config.online_similarity_decay,
            pmf_decay=stats_config.pmf_decay
        )


def rename_dict_keys(d: dict[str, Any], add_prefix):
    return {
        f'{add_prefix}{k}': d[k]
        for k in d
    }
