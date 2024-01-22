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
    n_steps: int

    def __init__(self, *shape: int):
        self.agg_value = 0. if not shape else np.zeros(shape)
        self.n_steps = 0

    def put(self, value: T, sdr: SparseSdr = None):
        if sdr is not None:
            self.agg_value[sdr] += value
        else:
            self.agg_value += value
        self.n_steps += 1

    def get(self) -> T:
        return safe_divide(self.agg_value, self.n_steps)

    def reset(self):
        if isinstance(self.agg_value, float):
            self.agg_value = 0.
        else:
            self.agg_value[:] = 0.
        self.n_steps = 0
