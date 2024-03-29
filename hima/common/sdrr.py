#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import numpy as np
import numpy.typing as npt

from hima.common.sdr import SparseSdr


@dataclass
class RateSdr:
    """
    Represent non-binary rate SDR aka Sparse Distributed Rate Representation
    (SDRR) stored in a compressed format:
        - sdr stores non-zero indices
        - values stores the corresponding non-zero values

    In most SDRR-related computations float values in [0, 1] are expected,
    representing relative rate or probability-like values — this is the main
    purpose of the structure.
    However, it may also be useful for other purposes, like to aggregate
    SDR-related int/float statistics. Therefore, the structure itself
    does NOT restrict the type or range of values.

    IMPORTANT: rates have a bit of obscure nature!
        - each rate is a value in [0, 1] representing independent rate of the neuron
        - it is relative to the neuron's maximum possible rate
        - we employ rate normalization: the output of a layer is always re-normalized in
            such way, that the most successful neuron will have a rate of 1. And the other
            winners will have their rates relative to the 1st.
        - therefore the following properties of RateSdr should be expected:
            - the absolute bit rate does not mean absolute strength of the recognition. Instead,
                it means how relatively good it recognizes the input pattern.
            - there is no way ATM to represent the certainty of recognition. In the future,
                it will be modelled with the size of RateSdr, such that the bigger size will
                mean less certainty and vice versa.
        TODO: draft; to be finalized after testing.

    NB: Be careful mutating values. By default, consider RateSdr objects as immutable.
    """
    sdr: SparseSdr
    values: npt.NDArray[int | float] | list[int | float] = None

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


def split_sdr_values(sdr: AnySparseSdr) -> tuple[SparseSdr, float | npt.NDArray[float]]:
    """Split SDR or Rate SDR into SDR and its rates."""
    if isinstance(sdr, RateSdr):
        return sdr.sdr, sdr.values

    return sdr, np.ones(len(sdr), dtype=float)
