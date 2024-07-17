#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from enum import Enum, auto
from typing import Union

import numpy as np
from numpy import typing as npt

from hima.common.utils import isnone

# ========================= Binary SDR ===============================

# SDR representation optimized for set operations. It is segregated to
# clarify, when a function work with this exact representation.
SetSdr = set[int]

# General sparse form SDR. In most cases, ndarray or list is expected.
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


# ========================= Rate SDR ===============================
class RateSdr:
    """
    Represent non-binary rate SDR aka Sparse Distributed Rate Representation
    (SDRR) stored in a compressed format:
        - sdr stores non-zero indices
        - values stores the corresponding non-zero values

    In most SDRR-related computations float values in [0, 1] are expected,
    representing relative rate or probability-like values — this is the main
    purpose of the structure.
    However, sometimes it may also be useful for other purposes, like to aggregate
    SDR-related int/float statistics. Therefore, the structure itself
    does NOT restrict the type or range of values.

    NB: Be careful mutating values. By default, consider RateSdr objects as immutable.
    """
    sdr: SparseSdr
    values: npt.NDArray[float]

    def __init__(self, sdr: SparseSdr, values: npt.NDArray[float] = None):
        if values is None:
            values = np.ones(len(sdr), dtype=float)

        self.sdr = sdr
        self.values = values

    def with_values(self, values):
        """Produce another RateSdr with new values over the same SDR."""
        return RateSdr(sdr=self.sdr, values=values)

    def reorder(self, ordering):
        """Reorder both SDR indices and their corresponding values."""
        self.sdr[:] = self.sdr[ordering]
        self.values[:] = self.values[ordering]

    # NB: doubtful decision to implement it as it could be misused
    # due to non-exact (=approx) equality check, or it could be
    # unintentionally overused slowing down the performance.
    def __eq__(self, other: RateSdr) -> bool:
        if self is other:
            return True
        if self.sdr is other.sdr and self.values is other.values:
            return True

        return (
                np.all(self.sdr == other.sdr)
                and np.allclose(self.values, other.values)
        )


# Aggregate type for functions that support both representations.
AnySparseSdr = Union[SparseSdr, RateSdr]
CompartmentsAnySparseSdr = dict[str, AnySparseSdr]


class OutputMode(Enum):
    BINARY = 1
    RATE = auto()


def unwrap_as_rate_sdr(sdr: AnySparseSdr) -> tuple[SparseSdr, float | npt.NDArray[float]]:
    """Split SDR or Rate SDR into SDR and its rates."""
    if isinstance(sdr, RateSdr):
        return sdr.sdr, sdr.values
    return sdr, np.ones(len(sdr), dtype=float)


def wrap_as_rate_sdr(sdr: AnySparseSdr) -> RateSdr:
    """Wrap SDR into Rate SDR."""
    if isinstance(sdr, RateSdr):
        return sdr
    return RateSdr(sdr)
