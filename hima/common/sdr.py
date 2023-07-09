#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Iterable, Union, Sequence

import numpy as np

SparseSdr = Union[Iterable[int], Sequence[int], np.ndarray]
DenseSdr = np.ndarray


def sparse_to_dense(indices: SparseSdr, total_size: int) -> DenseSdr:
    """Converts SDR from sparse representation to dense."""
    # TODO: test int vs int8
    dense_vector = np.zeros(total_size, dtype=np.int8)
    dense_vector[indices] = 1
    return dense_vector


def dense_to_sparse(dense_vector: DenseSdr) -> SparseSdr:
    return np.nonzero(dense_vector)[0]
