#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from typing import Optional


class BaseEncoder:
    n_vars: int
    n_states: int

    def encode(self, input_: np.ndarray, learn: bool) -> np.ndarray:
        """
            output should be a sdr
        """
        raise NotImplementedError

    def decode(
            self,
            input_: np.ndarray,
            learn: bool = False,
            correct: Optional[np.ndarray] = None
    ):
        return input_
