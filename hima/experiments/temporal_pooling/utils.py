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


class Scheduler:
    schedule: int
    max_value: int | None
    always_report_first: bool
    always_report_last: bool
    zero_based: bool

    def __init__(
            self, schedule: int = 1, max_value: int = None,
            always_report_first: bool = True, always_report_last: bool = True,
            zero_based: bool = True
    ):
        if always_report_last:
            assert max_value is not None

        self.schedule = schedule
        self.max_value = max_value
        self.always_report_first = always_report_first
        self.always_report_last = always_report_last
        self.zero_based = zero_based

    def scheduled(self, i: int) -> bool:
        if self.zero_based:
            i += 1

        if self.always_report_first and i == 1:
            return True

        if self.always_report_last and i == self.max_value:
            return True

        return i % self.schedule == 0
