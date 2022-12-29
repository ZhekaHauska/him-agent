#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
from numpy.random import Generator

from hima.common.sds import Sds


def sample_rf(sds: Sds, rf_sparsity: float, rng: Generator) -> np.ndarray:
    """Sample a random receptive field."""
    rf_size = int(sds.size * rf_sparsity)
    return rng.choice(sds.size, rf_size, replace=False)


def boosting(relative_rate: float | np.ndarray, log_k: float) -> float:
    # x = log(R / r)
    # B = exp(logK * x / |1 + x|)
    #   0 -> -inf -> exp(-logK) = 1 / K
    #   1 -> 0 -> exp(logK * 0) = 1
    #   +inf -> +inf -> exp(logK) = K
    x = np.log(relative_rate)
    return np.exp(log_k * x / np.abs(1 + x))
