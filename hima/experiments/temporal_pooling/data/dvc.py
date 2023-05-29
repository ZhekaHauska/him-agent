#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle
from dataclasses import dataclass

import numpy as np

from hima.common.sds import Sds
from hima.experiments.temporal_pooling.data.synthetic_sequences import Sequence


@dataclass
class DvcSdrs:
    n_elements: int
    sds: tuple[int, float]
    sdrs: list[np.ndarray]


class DvcSequences:
    sds: Sds
    n_sequences: int
    sequence_length: int
    sequences: list[Sequence]

    def __init__(
            self,
            n_sequences: int,
            sequence_length: int,
            filepath: str,
            seed: int,
            sequential: bool = True,
    ):
        self.n_sequences = n_sequences
        dataset = self._read_dataset(filepath)
        self.sds = Sds.make(dataset.sds)
        sdrs = dataset.sdrs
        sdrs = sdrs[3000:-4000]
        ds_size = len(sdrs)

        self.sequences = []
        i_sequence = 0
        rng = np.random.default_rng(seed)
        while len(self.sequences) < n_sequences or i_sequence > 100_000:
            if sequential:
                start = i_sequence*sequence_length % ds_size
            else:
                start = rng.integers(ds_size - sequence_length)
            end = start + sequence_length
            i_sequence += 1
            if end >= ds_size:
                continue

            self.sequences.append(
                Sequence(id=len(self.sequences), seq=sdrs[start:end])
            )

    def __iter__(self):
        return iter(self.sequences)

    @staticmethod
    def _read_dataset(filepath: str) -> DvcSdrs:
        with open(filepath, mode='rb') as f:
            dataset = pickle.load(f)
        return DvcSdrs(**dataset)

