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
    ):
        self.n_sequences = n_sequences
        dataset = self._read_dataset(filepath)
        self.sds = Sds.make(dataset.sds)
        sdrs = dataset.sdrs
        sdrs = sdrs[3000:-4000]

        self.sequences = [
            Sequence(
                id=i_sequence,
                seq=sdrs[i_sequence*sequence_length:(i_sequence+1)*sequence_length]
            )
            for i_sequence in range(n_sequences)
        ]

    def __iter__(self):
        return iter(self.sequences)

    @staticmethod
    def _read_dataset(filepath: str) -> DvcSdrs:
        with open(filepath, mode='rb') as f:
            dataset = pickle.load(f)
        return DvcSdrs(**dataset)

