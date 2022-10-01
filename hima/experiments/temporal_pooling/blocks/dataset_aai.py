#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle
from typing import Iterator, Any

import numpy as np
from htm.bindings.sdr import SDR

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import isnone
from hima.experiments.temporal_pooling.blocks.base_block_stats import BlockStats
from hima.experiments.temporal_pooling.sdr_seq_cross_stats import OfflineElementwiseSimilarityMatrix
from hima.experiments.temporal_pooling.sdr_seq_stats import SdrSequenceStats
from hima.experiments.temporal_pooling.stats_config import StatsMetricsConfig


class RoomObservationSequence:
    """Sequence of observations."""
    id: int

    _observations: list[SparseSdr]

    def __init__(self, id_: int, observations):
        self.id = id_
        self._observations = observations

    def __iter__(self) -> Iterator[SparseSdr]:
        return iter(self._observations)

    def __len__(self):
        return len(self._observations)


class AnimalAiDatasetBlockStats(BlockStats):
    n_sequences: int
    observations_sds: Sds
    observation_sequences: list[list[set[int]]]
    sds_stats: SdrSequenceStats
    cross_stats: OfflineElementwiseSimilarityMatrix
    # cross_stats: SdrSequencesOfflineCrossStats

    def __init__(
            self, observation_sequences: list[RoomObservationSequence], sds: Sds,
            stats_config: StatsMetricsConfig
    ):
        super(AnimalAiDatasetBlockStats, self).__init__(output_sds=sds)

        self.n_sequences = len(observation_sequences)
        self.observation_sequences = [
            [set(obs) for obs in seq]
            for seq in observation_sequences
        ]
        self.observations_sds = sds
        self.sds_stats = SdrSequenceStats(self.observations_sds)
        self.cross_stats = OfflineElementwiseSimilarityMatrix(
            sequences=self.observation_sequences,
            normalization=stats_config.normalization,
            discount=stats_config.prefix_similarity_discount,
            symmetrical=False
        )
        # self.cross_stats = SdrSequencesOfflineCrossStats(
        #     sequences=self.observation_sequences, sds=self.observations_sds,
        #     prefix_algorithm='prefix.point_similarity',
        #     prefix_discount=stats_config.prefix_similarity_discount,
        #     unbias_func=stats_config.normalization_unbias
        # )

    @staticmethod
    def distance_based_similarities() -> np.ndarray:
        return np.asarray(
            [[1, 0.91, 0.94, 0.94, 0.625],
             [0.91, 1, 0.94, 0.94, 0.68],
             [0.94, 0.94, 1, 0.91, 0.69],
             [0.94, 0.94, 0.91, 1, 0.62],
             [0.625, 0.68, 0.69, 0.62, 1]]
        )

    def reset(self):
        self.sds_stats = SdrSequenceStats(self.observations_sds)

    def update(self, current_output_sdr: SparseSdr):
        self.sds_stats.update(current_output_sdr)

    def step_metrics(self) -> dict[str, Any]:
        return self.sds_stats.step_metrics()

    def final_metrics(self) -> dict[str, Any]:
        return self.cross_stats.final_metrics()


class AnimalAiDatasetBlock:
    id: int
    name: str

    n_sequences: int
    n_observations_per_sequence: int
    output_sds: Sds

    stats: AnimalAiDatasetBlockStats

    _observation_sequences: list[RoomObservationSequence]

    def __init__(
            self, sds: Sds, observation_sequences: list[RoomObservationSequence],
            stats_config: StatsMetricsConfig
    ):
        self.output_sds = sds
        self._observation_sequences = observation_sequences
        self.n_sequences = len(self._observation_sequences)
        self.n_observations_per_sequence = len(self._observation_sequences[0])

        self.stats = AnimalAiDatasetBlockStats(
            self._observation_sequences, self.output_sds, stats_config
        )

    @property
    def tag(self) -> str:
        return f'{self.id}_in'

    def __iter__(self) -> Iterator[RoomObservationSequence]:
        return iter(self._observation_sequences)

    def update_stats(self, observation: SparseSdr):
        self.stats.update(observation)

    def reset_stats(self):
        self.stats.reset()


class AAIRotationsGenerator:
    TDataset = list[list[SDR]]

    n_sequences: int
    n_rotations: int
    v1_output_sequences: TDataset
    v1_output_sds: Sds

    def __init__(
            self, sds: Sds.TShortNotation, filepath: str
    ):
        self.v1_output_sequences = self.restore_dataset(filepath)
        self.v1_output_sds = Sds(short_notation=sds)

        self.n_sequences = len(self.v1_output_sequences)
        self.n_rotations = len(self.v1_output_sequences[0])

    def generate_data(
            self, stats_config: StatsMetricsConfig,
            n_sequences: int = None, n_observations_per_sequence: int = None
    ):
        n_sequences = isnone(n_sequences, self.n_sequences)
        n_rotations = isnone(n_observations_per_sequence, self.n_rotations)

        # NB: take only requested N of sequences and rotations per sequence
        observation_sequences = [
            RoomObservationSequence(
                id_=ind,
                observations=[
                    np.array(obs.sparse, copy=True)
                    for obs in observations[:n_rotations]
                ]
            )
            for ind, observations in enumerate(self.v1_output_sequences[:n_sequences])
        ]
        return AnimalAiDatasetBlock(
            sds=self.v1_output_sds,
            observation_sequences=observation_sequences,
            stats_config=stats_config
        )

    @staticmethod
    def restore_dataset(filepath: str) -> TDataset:
        with open(filepath, 'rb') as dataset_io:
            return pickle.load(dataset_io)
