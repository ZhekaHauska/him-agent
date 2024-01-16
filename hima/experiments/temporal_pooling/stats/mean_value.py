#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np


class MeanValue:
    value: list[float]

    def __init__(self):
        self.value = []

    def put(self, x: float):
        self.value.append(x)

    def get(self) -> float:
        return np.mean(self.value)

    def reset(self):
        self.value.clear()
