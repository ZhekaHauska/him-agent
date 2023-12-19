#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt

from hima.common.utils import isnone

# SDR representation optimized for set operations. It is segregated to
# clarify, when a function work with this exact representation.
SetSdr = set[int]

# General sparse form SDR. In most cases, list or ndarray is expected.
SparseSdr = Union[list[int], npt.NDArray[int], SetSdr]

# Dense SDR form. Could be a list too, but in general it's ndarray.
DenseSdr = npt.NDArray[Union[int, float]]


def sparse_to_dense(
        sdr: SparseSdr,
        size: int | tuple | DenseSdr = None,
        shape: int | tuple | DenseSdr = None,
        dtype=float,
        like: DenseSdr = None
) -> DenseSdr:
    """
    Converts SDR from sparse representation to dense.

    Size, shape and dtype define resulting dense vector params.
    The size should be at least inducible (from shape or like).
    The shape default is 1-D, dtype: float.

    Like param is a shorthand, when you have an array with all three params set correctly.
    Like param overwrites all others!
    """

    if like is not None:
        shape, size, dtype = like.shape, like.size, like.dtype
    else:
        if isinstance(size, np.ndarray):
            size = size.size
        if isinstance(shape, np.ndarray):
            shape = shape.shape

        # -1 for reshape means flatten.
        # It is also invalid size, which we need here for the unset shape case.
        shape = isnone(shape, -1)
        size = isnone(size, np.prod(shape))

    dense_vector = np.zeros(size, dtype=dtype)
    dense_vector[sdr] = 1
    return dense_vector.reshape(shape)


def dense_to_sparse(dense_vector: DenseSdr) -> SparseSdr:
    return np.flatnonzero(dense_vector)
