#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle
from dataclasses import dataclass

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
    ):
        self.n_sequences = n_sequences
        dataset = self._read_dataset(filepath)
        ds_set = set(dataset)
        alphabet_size = len(ds_set)
        mapping = {
            c: i
            for i, c in enumerate(ds_set)
        }
        dataset = np.array([mapping[c] for c in dataset])

        print(len(dataset), alphabet_size)
        self.encoder = global_config.resolve_object(encoder, n_values=alphabet_size)
        self.sds = self.encoder.output_sds

        self.sequences = [
            Sequence(id=i, seq=self.encoder.encode(dataset))
            for i in range(n_sequences)
        ]

    def __iter__(self):
        return iter(self.sequences)

    @staticmethod
    def _read_dataset(filepath: str):
        with open(filepath, mode='r') as f:
            text = str.join('\n', f.readlines())

        text = list(text)
        return text

