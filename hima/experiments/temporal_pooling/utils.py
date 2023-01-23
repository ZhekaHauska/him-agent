#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np

from hima.common.config.values import resolve_value


def resolve_random_seed(seed: int | None) -> int:
    seed = resolve_value(seed)
    if seed is None:
        # generate a random seed
        return np.random.default_rng().integers(10000)
    return seed


def scheduled(
        i: int, schedule: int = 1,
        always_report_first: bool = True, always_report_last: bool = True, i_max: int = None
):
    if always_report_first and i == 0:
        return True
    if always_report_last and i + 1 == i_max:
        return True
    if (i + 1) % schedule == 0:
        return True
    return False
