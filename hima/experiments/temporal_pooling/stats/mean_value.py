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
from hima.experiments.temporal_pooling.stp.sp_utils import (
    tick, make_repeating_counter, is_infinite
)

T = TypeVar('T', float, npt.NDArray[float])


# TODO: support tracking STD


class LearningRateParam:
    """Allows specification of learning rate as learning rate, decay or learning window."""
    lr: float
    decay: float
    window: int

    def __init__(self, lr: float = None, decay: float = None, window: int = None):
        self.lr, self.decay, self.window = self.resolve_params(lr, decay, window)

    def n_step_lr(self, n_steps: int) -> float:
        return 1 - self.n_step_decay(n_steps)

    def n_step_decay(self, n_steps: int) -> float:
        return self.decay ** n_steps

    @staticmethod
    def resolve_params(lr: float = None, decay: float = None, window: int = None):
        if lr is not None:
            decay = 1 - lr
            window = int(safe_divide(1, lr))
        elif decay is not None:
            lr = 1 - decay
            window = int(safe_divide(1, lr))
        elif window is not None:
            lr = safe_divide(1, window)
            decay = 1 - lr
        else:
            raise ValueError("No learning rate specified.")
        return lr, decay, window


class MeanValue(Generic[T]):
    """
    Implements approximate exponential moving average.

    "Approximate" here means that the exponential decay is applied only periodically
    for faster performance. Decay application itself does not change the average value,
    but re-weights the past values over the future values. This means that after a
    decay, there is a small learning rate "bump" (and therefore a mean approximation bias)
    that is expected to be negligible if decay >> 0 (= learning rate << 1).
    I advise to use learning rate < 0.05.
    """

    agg_value: T
    n_steps: float

    is_array: bool
    lr: LearningRateParam

    def __init__(
            self, *, lr: LearningRateParam, size: int = None, auto_decay: bool = True,
            initial_value: float = 0.
    ):
        self.is_array = size is not None
        self.lr = lr
        self.safe_window = self.get_safe_window()

        self.n_steps = self.safe_window
        self.initial_value = self.n_steps * initial_value
        if self.is_array:
            self.agg_value = np.full(size, fill_value=self.initial_value)
        else:
            self.agg_value = self.initial_value

        self.countdown = make_repeating_counter(self.safe_window if auto_decay else None)

    def put(self, value: T | float, sdr: SparseSdr = None):
        if sdr is not None:
            # only for array: sliced update
            self.agg_value[sdr] += value
        else:
            # full update: array or scalar
            self.agg_value += value

        self.n_steps += 1.0

        # if auto-decay is on (counter is non-infinite), apply decay periodically
        is_now, self.countdown = tick(self.countdown)
        if is_now:
            self.reset()

    def get(self) -> T:
        return safe_divide(self.agg_value, self.n_steps)

    def reset(self, hard: bool = False):
        # it is expected to apply reset/decay only periodically, not after every
        # step to improve performance
        auto_decay = not is_infinite(self.countdown)
        if auto_decay:
            n_steps = self.countdown[1]
        else:
            # -1 - (-n_steps - 1) = n_steps
            n_steps = self.countdown[1] - self.countdown[0]
            self.countdown = make_repeating_counter(None)

        if hard:
            self.n_steps = self.safe_window
            if self.is_array:
                self.agg_value = np.full_like(self.agg_value, fill_value=self.initial_value)
            else:
                self.agg_value = self.initial_value
        else:
            decay = self.lr.n_step_decay(n_steps)
            self.agg_value *= decay
            self.n_steps *= decay

    def get_safe_window(self):
        safe_window = np.sqrt(self.lr.window)
        return max(1, int(safe_window))
