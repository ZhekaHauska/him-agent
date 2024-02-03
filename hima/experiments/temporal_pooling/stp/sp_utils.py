#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Optional

import numpy as np
from numba import jit
from numpy.random import Generator


def gather_rows(arr_2d: np.ndarray, indices: np.ndarray):
    if indices.ndim == 1:
        indices = np.expand_dims(indices, axis=1)

    # ==> arr_2d[i, indices[i]]
    return np.take_along_axis(
        arr=arr_2d, indices=indices, axis=1
    )


def sample_rf(ff_size: int, rf_sparsity: float, rng: Generator) -> np.ndarray:
    """Sample a random receptive field."""
    rf_size = int(ff_size * rf_sparsity)
    return rng.choice(ff_size, rf_size, replace=False)


def sample_for_each_neuron(
        rng: Generator, n_neurons,
        set_size: int | np.ndarray, sample_size: int,
        probs_2d: np.ndarray = None
) -> np.ndarray:
    return np.array([
        rng.choice(
            set_size, size=sample_size, replace=False,
            p=probs_2d[neuron] if probs_2d is not None else None
        )
        for neuron in range(n_neurons)
    ])


def boosting(relative_rate: float | np.ndarray, k: float, softness: float = 3.0) -> float:
    # relative rate: rate / R_target
    # x = -log(relative_rate)
    #   0 1 +inf  -> +inf 0 -inf
    x = -np.log(relative_rate)

    # relative_rate -> x -> B:
    #   0 -> +inf -> K^tanh(+inf) = K
    #   1 -> 0 -> K^tanh(0) = 1
    #   +inf -> -inf -> K^tanh(-inf) = 1 / K
    # higher softness just makes the sigmoid curve smoother; default value is empirically optimized
    return np.power(k + 1, np.tanh(x / softness))


RepeatingCountdown = tuple[int, int]


@jit(inline='always')
def make_repeating_counter(ticks: int) -> RepeatingCountdown:
    if ticks is None:
        ticks = -1
    return ticks, ticks


@jit(inline='always')
def is_infinite(countdown: RepeatingCountdown) -> bool:
    return countdown[1] == -1


@jit
def tick(countdown: RepeatingCountdown) -> tuple[bool, RepeatingCountdown]:
    """Return True if the countdown has reached zero."""
    ticks_left, initial_ticks = countdown[0] - 1, countdown[1]
    if ticks_left == 0:
        return True, make_repeating_counter(initial_ticks)
    return False, (ticks_left, initial_ticks)
