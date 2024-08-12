#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
from numba import jit
from numpy import typing as npt
from numpy.random import Generator

from hima.common.sdr import OutputMode
from hima.common.utils import safe_divide


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


def boosting(
        relative_rate: float | npt.NDArray[float], k: float | npt.NDArray[float],
        softness: float = 3.0
) -> float:
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


@jit
def nb_choice(rng, p):
    """Choose a sample from N values with weights p (they could be non-normalized)."""
    # Get cumulative weights
    acc_w = np.cumsum(p)
    # Total of weights
    mx_w = acc_w[-1]
    r = mx_w * rng.random()
    # Get corresponding index
    ind = np.searchsorted(acc_w, r, side='right')
    return ind


@jit()
def nb_choice_k(
        rng: Generator, k: int, weights: npt.NDArray[np.float64] = None, n: int = None,
        replace: bool = False, cache: npt.NDArray[np.bool_] = None
):
    """Choose k samples from max_n values, with optional weights and replacement."""
    acc_w = np.cumsum(weights) if weights is not None else np.arange(0, n, 1, dtype=np.float64)
    # Total of weights
    mx_w = acc_w[-1]
    # result
    result = np.full(k, -1, dtype=np.int64)
    if not replace and cache is None and n is not None:
        cache = np.zeros(n, dtype=np.bool_)

    i, j = 0, 0
    timelimit = 1_000_000
    while i < k and j < timelimit:
        j += 1
        r = mx_w * rng.random()
        ind = np.searchsorted(acc_w, r, side='right')

        if not replace and cache[ind]:
            continue
        else:
            result[i] = ind
            if not replace:
                cache[ind] = True
            i += 1

    if j >= timelimit:
        raise ValueError('Infinite loop in nb_choice_k')

    return result


RepeatingCountdown = tuple[int, int]


@jit(cache=True, inline='always')
def make_repeating_counter(ticks: int | None) -> RepeatingCountdown:
    if ticks is None:
        ticks = -1
    return ticks, ticks


@jit(cache=True, inline='always')
def is_infinite(countdown: RepeatingCountdown) -> bool:
    return countdown[1] == -1


@jit
def tick(countdown: RepeatingCountdown) -> tuple[bool, RepeatingCountdown]:
    """Return True if the countdown has reached zero."""
    ticks_left, initial_ticks = countdown[0] - 1, countdown[1]
    if ticks_left == 0:
        return True, make_repeating_counter(initial_ticks)
    return False, (ticks_left, initial_ticks)


def normalize_weights(weights: npt.NDArray[float]):
    normalizer = np.abs(weights)
    if weights.ndim == 2:
        normalizer = normalizer.sum(axis=1, keepdims=True)
    else:
        normalizer = normalizer.sum()
    return np.clip(weights / normalizer, 0., 1)


def define_winners(potentials, winners, output_mode, normalize_rates, strongest_winner=None):
    winners = winners[potentials[winners] > 0]

    if output_mode == OutputMode.RATE:
        winners_value = potentials[winners].copy()
        if normalize_rates:
            winners_value = safe_divide(winners_value, potentials[strongest_winner])
    else:
        winners_value = 1.0

    return winners, winners_value
