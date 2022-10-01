#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

class BalancingParam:
    value: float
    min_value: float
    max_value: float
    delta: float
    negative_delta_rate: float

    def __init__(
            self, initial_value: float, min_value: float, max_value: float,
            delta: float, negative_delta_rate: float,
    ):
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.delta = delta
        self.negative_delta_rate = negative_delta_rate

    def balance(self, increase: bool = True):
        if increase:
            new_value = self.value + self.delta
        else:
            new_value = self.value - self.delta * self.negative_delta_rate

        if self.min_value < new_value < self.max_value:
            self.value = new_value
