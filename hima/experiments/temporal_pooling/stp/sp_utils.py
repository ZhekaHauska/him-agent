#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
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
        set_size: int, sample_size: int,
        probs_2d: np.ndarray = None
) -> np.ndarray:
    samples = np.array([
        rng.choice(
            set_size, size=sample_size, replace=False,
            p=probs_2d[neuron] if probs_2d is not None else None
        )
        for neuron in range(n_neurons)
    ])
    # sort sampled indices. it's potentially a speed-up.
    samples.sort(axis=1)
    return samples


def boosting(relative_rate: float | np.ndarray, log_k: float) -> float:
    # x = log(R / r)
    # B = exp(logK * x / |1 + x|)
    #   0 -> -inf -> exp(-logK) = 1 / K
    #   1 -> 0 -> exp(logK * 0) = 1
    #   +inf -> +inf -> exp(logK) = K
    x = np.log(relative_rate)
    return np.exp(log_k * x / np.abs(1 + x))
