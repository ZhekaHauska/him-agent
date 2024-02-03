#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import TypeVar, Generic

import numpy as np
import numpy.typing as npt

from hima.common.sdr import SparseSdr
from hima.common.utils import safe_divide

T = TypeVar('T', float, npt.NDArray[float])


class MeanValue(Generic[T]):
    agg_value: T
    n_steps: float

    avg_mass: float

    is_array: bool
    exp_decay: float

    def __init__(self, *, size: int = None, exp_decay: float = 0.5):
        self.is_array = size is not None
        self.exp_decay = exp_decay

        self.agg_value = np.zeros(size) if self.is_array else 0.
        self.n_steps = 0.

    def put(self, value: T | float, sdr: SparseSdr = None):
        if sdr is not None:
            # only for array: sliced update
            self.agg_value[sdr] += value
        else:
            # full update: array or scalar
            self.agg_value += value

        self.n_steps += 1.0

    def get(self) -> T:
        return safe_divide(self.agg_value, self.n_steps)

    def reset(self, hard: bool = False):
        # it is expected to apply reset/decay only periodically, not after every
        # step to improve performance
        decay = self.exp_decay if not hard else 0.

        self.agg_value *= decay
        self.n_steps *= decay
