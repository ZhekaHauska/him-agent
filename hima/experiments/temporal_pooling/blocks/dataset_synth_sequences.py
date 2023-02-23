#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import dataclasses
from typing import Iterator, Any

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling.data.synthetic_sequences import (
    generate_synthetic_sequences, generate_synthetic_single_element_sequences
)


@dataclasses.dataclass
class Sequence:
    """Sequence of SDRs."""
    id: int
    seq: list[SparseSdr]

    def __iter__(self) -> Iterator[SparseSdr]:
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)


class SyntheticSequences:
    alphabet_size: int
    encoder: Any
    n_sequences: int
    sequence_length: int
    sequences: list[Sequence]

    def __init__(
            self, global_config: GlobalConfig,
            n_sequences: int,
            sequence_length: int,
            alphabet_size: int,
            encoder: TConfig,
            **generator_kwargs: TConfig
    ):
        self.alphabet_size = alphabet_size
        self.encoder = global_config.resolve_object(encoder, n_values=alphabet_size)

        if sequence_length > 1:
            sequences = generate_synthetic_sequences(
                n_sequences=n_sequences, sequence_length=sequence_length,
                alphabet_size=alphabet_size, **generator_kwargs
            )
        else:
            sequences = generate_synthetic_single_element_sequences(
                n_sequences=n_sequences, alphabet_size=alphabet_size,
                **generator_kwargs
            )

        self.n_sequences = n_sequences
        self.sequences = [
            Sequence(id=i_sequence, seq=self.encoder.encode(sequence))
            for i_sequence, sequence in enumerate(sequences)
        ]

