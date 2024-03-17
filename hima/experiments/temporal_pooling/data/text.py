#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from pathlib import Path

import numpy as np

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.data.synthetic_sequences import Sequence


class TextSequences:
    sds: Sds
    n_sequences: int
    sequence_length: int
    sequences: list[Sequence]

    def __init__(
            self, global_config: GlobalConfig,
            filepath: str,
            n_sequences: int,
            sequence_length: int,
            encoder: TConfig,
            seed: int,
            sequential: bool = True,
            max_size_hint: int = None
    ):
        self.n_sequences = n_sequences
        dataset, alphabet_size, mapping = self._read_dataset(filepath, max_size_hint)
        ds_size = len(dataset)
        print(f'Text dataset size: {ds_size} | Alphabet size: {alphabet_size}')

        self.encoder = global_config.resolve_object(encoder, n_values=alphabet_size)
        self.sds = self.encoder.output_sds
        self.char_mapping = mapping

        self.sequences = []
        i_sequence = 0
        rng = np.random.default_rng(seed)
        while len(self.sequences) < n_sequences:
            if sequential:
                start = i_sequence*sequence_length % ds_size
            else:
                start = rng.integers(ds_size - sequence_length)

            end = start + sequence_length
            i_sequence += 1
            if end >= ds_size:
                continue

            encoded_seq = self.encoder.encode(dataset[start:end])
            seq = Sequence(id=len(self.sequences), seq=encoded_seq)
            self.sequences.append(seq)

    def __iter__(self):
        return iter(self.sequences)

    @staticmethod
    def _read_dataset(filepath: str, max_size_hint: int = None):
        filepath = Path(filepath)
        filepath = filepath.expanduser()
        with open(filepath, mode='r', encoding='utf-8') as f:
            text = f.read(max_size_hint)

        text = list(text)
        chars = set(text)
        mapping = {
            c: i
            for i, c in enumerate(sorted(chars))
        }

        alphabet_size = len(chars)
        dataset = np.array([mapping[c] for c in text])
        return dataset, alphabet_size, mapping

