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

    track_avg_mass: bool
    avg_mass: float

    is_array: bool
    exp_decay: float

    def __init__(self, *, size: int = None, track_avg_mass: bool = False, exp_decay: float = 0.):
        self.is_array = size is not None
        self.track_avg_mass = track_avg_mass
        self.exp_decay = exp_decay

        self.agg_value = np.zeros(size) if self.is_array else 0.
        self.n_steps = 0.
        self.avg_mass = 0.

    def put(self, value: T, sdr: SparseSdr = None, avg_mass: float = None):
        if sdr is not None:
            # only for array: sliced update
            self.agg_value[sdr] += value
        else:
            # full update: array or scalar
            self.agg_value += value

        self.n_steps += 1.0

        if self.track_avg_mass:
            if avg_mass is None:
                # only for array: update by average mass
                self.avg_mass += np.mean(value)
            else:
                # for array or scalar
                self.avg_mass += avg_mass

    def get(self, with_avg_mass: bool = False) -> T:
        if with_avg_mass:
            return safe_divide(self.agg_value, self.avg_mass)
        return safe_divide(self.agg_value, self.n_steps)

    def decay(self):
        # it is expected to apply decay only periodically, not after every
        # step to improve performance
        self.agg_value *= self.exp_decay
        self.n_steps *= self.exp_decay
        self.avg_mass *= self.exp_decay

    def reset(self):
        if self.is_array:
            self.agg_value.fill(0.)
        else:
            self.agg_value = 0.

        self.n_steps = 0.
        self.avg_mass = 0.


class ScalarMeanValue:
    agg_value: float
    n_steps: float

    exp_decay: float

    def __init__(self, exp_decay: float = 0.):
        self.exp_decay = exp_decay

        self.agg_value = 0.
        self.n_steps = 0.

    def put(self, value: float, step_size: float = 1.0):
        self.agg_value += value
        self.n_steps += step_size

    def get(self) -> float:
        return safe_divide(self.agg_value, self.n_steps)

    def decay(self):
        # it is expected to apply decay only periodically, not after every
        # step to improve performance
        self.agg_value *= self.exp_decay
        self.n_steps *= self.exp_decay

    def reset(self):
        self.agg_value = 0.
        self.n_steps = 0.


class ArrayMeanValue:
    agg_value: npt.NDArray[float]
    n_steps: float

    step_by_avg_mass: bool
    exp_decay: float

    def __init__(self, size: int, step_by_avg_mass: bool = False, exp_decay: float = 0.):
        self.step_by_avg_mass = step_by_avg_mass
        self.exp_decay = exp_decay

        self.agg_value = np.zeros(size, dtype=float)
        self.n_steps = 0.

    def put(
            self, value: npt.NDArray[float], sdr: SparseSdr = None,
            step_size: float = 1.0
    ):
        if sdr is not None:
            # sliced update
            self.agg_value[sdr] += value
        else:
            self.agg_value += value

        if self.step_by_avg_mass:
            self.n_steps += np.mean(value)
        else:
            self.n_steps += step_size

    def get(self) -> T:
        return safe_divide(self.agg_value, self.n_steps)

    def decay(self):
        # it is expected to apply decay only periodically, not after every
        # step to improve performance
        self.agg_value *= self.exp_decay
        self.n_steps *= self.exp_decay

    def reset(self):
        self.agg_value.fill(0.)
        self.n_steps = 0.
