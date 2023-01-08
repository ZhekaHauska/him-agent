#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle
from typing import Iterator

import numpy as np
from htm.bindings.sdr import SDR

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds, TSdsShortNotation
from hima.common.utils import isnone


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


class AnimalAiDatasetBlock:
    id: int
    name: str

    n_sequences: int
    n_observations_per_sequence: int
    output_sds: Sds

    _observation_sequences: list[RoomObservationSequence]

    def __init__(self, sds: Sds, observation_sequences: list[RoomObservationSequence]):
        self.output_sds = sds
        self._observation_sequences = observation_sequences
        self.n_sequences = len(self._observation_sequences)
        self.n_observations_per_sequence = len(self._observation_sequences[0])

    @property
    def tag(self) -> str:
        return f'{self.id}_in'

    def __iter__(self) -> Iterator[RoomObservationSequence]:
        return iter(self._observation_sequences)


class AAIRotationsGenerator:
    TDataset = list[list[SDR]]

    n_sequences: int
    n_rotations: int
    v1_output_sequences: TDataset
    v1_output_sds: Sds

    def __init__(self, sds: TSdsShortNotation, filepath: str):
        self.v1_output_sequences = self.restore_dataset(filepath)
        self.v1_output_sds = Sds(short_notation=sds)

        self.n_sequences = len(self.v1_output_sequences)
        self.n_rotations = len(self.v1_output_sequences[0])

    def generate_data(self, n_sequences: int = None, n_observations_per_sequence: int = None):
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
        )

    @staticmethod
    def restore_dataset(filepath: str) -> TDataset:
        with open(filepath, 'rb') as dataset_io:
            return pickle.load(dataset_io)
