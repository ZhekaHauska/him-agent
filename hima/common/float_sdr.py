#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hima.common.sdr import SparseSdr


@dataclass
class FloatSparseSdr:
    """
    Represent non-binary SDR stored in a compressed format:
        - sdr stores non-zero indices
        - values stores the corresponding non-zero values
    """
    sdr: SparseSdr
    values: np.ndarray = None

    # NB: doubtful decision to implement it as it could be misused
    # due to approx equality check or could be unintentionally overused
    def __eq__(self, other: FloatSparseSdr) -> bool:
        if self is other:
            return True
        if self.sdr is other.sdr and self.values is other.values:
            return True

        return (
            np.all(self.sdr == other.sdr)
            and np.allclose(self.values, other.values)
        )
