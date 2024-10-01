#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from numba.experimental import jitclass


@jitclass
class Scheduler:
    remain: int
    window: int

    def __init__(self, window: int | None):
        if window is None:
            window = 0

        self.window = window
        self.remain = self.window

    @property
    def is_infinite(self):
        return self.window == 0

    @property
    def ticks_passed(self):
        """
        Returns the number of ticks passed since the last reset.
        NB: for infinite scheduler, this is the total ticks passed.
        """
        return self.window - self.remain

    def tick(self, n: int = 1) -> int:
        self.remain -= n

        if self.is_infinite:
            return 0

        n_events = 0
        while self.remain <= 0:
            self.remain += self.window
            n_events += 1
        return n_events

    def reset(self):
        self.remain = self.window
