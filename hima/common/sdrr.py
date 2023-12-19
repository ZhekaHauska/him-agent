#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from dataclasses import dataclass
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


AnySparseSdr = Union[SparseSdr, RateSdr]
