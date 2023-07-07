#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np

EPS = 1e-24
INT_TYPE = "int64"
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
_TIE_BREAKER_FACTOR = 1e-24


def softmax(x, beta=1.0):
    e_x = np.exp(beta * (x - x.mean()))
    return e_x / e_x.sum()


def normalize(x, default_values=None):
    norm = x.sum(axis=-1)
    mask = norm == 0

    if default_values is None:
        default_values = np.ones_like(x)

    x[mask] = default_values[mask]
    norm[mask] = x[mask].sum(axis=-1)
    return x / norm.reshape((-1, 1))


def sample_categorical_variables(probs, rng: np.random.Generator):
    assert np.allclose(probs.sum(axis=-1), 1)

    gammas = rng.uniform(size=probs.shape[0]).reshape((-1, 1))

    dist = np.cumsum(probs, axis=-1)

    ubounds = dist
    lbounds = np.zeros_like(dist)
    lbounds[:, 1:] = dist[:, :-1]

    cond = (gammas >= lbounds) & (gammas < ubounds)

    states = np.zeros_like(probs) + np.arange(probs.shape[1])

    samples = states[cond]

    return samples
