#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Iterator

import numpy as np
from numpy.random import Generator

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.blocks.graph import Block
from hima.experiments.temporal_pooling.data.synthetic_sequences import (
    generate_synthetic_sequences, generate_synthetic_single_element_sequences
)


class Sequence:
    """Sequence of SDRs."""
    id: int
    raw_sequence: list[SparseSdr]

    def __init__(self, id_: int, sequence):
        self.id = id_
        self.raw_sequence = sequence

    def __iter__(self) -> Iterator[dict]:
        for sdr in self.raw_sequence:
            yield dict(input=sdr)

    def __len__(self):
        return len(self.raw_sequence)


class SyntheticSequencesDatasetBlock(Block):
    OUTPUT = 'output'

    family = "generator"
    supported_streams = {OUTPUT}

    n_values: int
    _sequences: list[Sequence]

    def __init__(
            self, id: int, name: str, n_values: int, sequences: list[Sequence],
            values_sds: Sds
    ):
        super(SyntheticSequencesDatasetBlock, self).__init__(id, name)

        self.n_values = n_values
        self._sequences = sequences
        self.register_stream(self.OUTPUT).resolve_sds(values_sds)

    def reset(self, **kwargs):
        super(SyntheticSequencesDatasetBlock, self).reset(**kwargs)

    def build(self):
        pass

    def __iter__(self) -> Iterator[Sequence]:
        return iter(self._sequences)

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        ...

    def reset_stats(self):
        ...


class SyntheticSequencesGenerator:
    family = "generator"

    sequence_length: int
    n_values: int
    values_sds: Sds

    sequence_similarity: float
    sequence_similarity_std: float

    seed: int
    _rng: Generator

    def __init__(
            self, global_config: GlobalConfig,
            sequence_length: int, n_values: int, active_size: int,
            value_encoder: TConfig,
            sequence_similarity: float,
            seed: int,
            sequence_similarity_std: float = 0.
    ):
        self.sequence_length = sequence_length
        self.n_values = n_values
        self.value_encoder = global_config.resolve_object(
            value_encoder,
            n_values=self.n_values,
            active_size=active_size,
            seed=seed
        )
        self.values_sds = self.value_encoder.output_sds

        self.sequence_similarity = sequence_similarity
        self.sequence_similarity_std = sequence_similarity_std

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_sequences(self, n_sequences: int) -> list[Sequence]:
        values_encoding = [self.value_encoder.encode(x) for x in range(self.n_values)]

        if self.sequence_length > 1:
            sequences = generate_synthetic_sequences(
                n_sequences=n_sequences,
                sequence_length=self.sequence_length,
                n_values=self.n_values,
                seed=self.seed,
                sequence_similarity=self.sequence_similarity,
                sequence_similarity_std=self.sequence_similarity_std,
            )
        else:
            sequences = generate_synthetic_single_element_sequences(
                n_sequences=n_sequences, n_values=self.n_values, seed=self.seed
            )

        encoded_sequences = [
            Sequence(
                id_=i_sequence,
                sequence=[values_encoding[x] for x in sequence]
            )
            for i_sequence, sequence in enumerate(sequences)
        ]

        return encoded_sequences

    def make_block(
            self, block_id: int, block_name: str, sequences: list[Sequence]
    ) -> SyntheticSequencesDatasetBlock:
        return SyntheticSequencesDatasetBlock(
            id=block_id, name=block_name,
            n_values=self.n_values,
            sequences=sequences,
            values_sds=self.values_sds
        )
