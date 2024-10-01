#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from dhtm.common.utils import standardize, lin_sum


class SSValue:
    """
        Smoothed Standard Value
    """
    def __init__(
            self,
            lr,
            lr_norm,
            mean_ini=0.0,
            std_ini=1.0,
    ):
        self.lr = lr
        self.lr_norm = lr_norm

        self.mean = mean_ini
        self.std = std_ini
        self.disp = std_ini**2
        self.current_value = mean_ini

    def update(self, value):
        self.current_value = lin_sum(self.current_value, self.lr, value)
        self.mean = lin_sum(self.mean, self.lr_norm, value)
        self.disp = lin_sum(
            self.disp,
            self.lr_norm,
            (value - self.mean)**2
        )
        self.std = self.disp**0.5

    @property
    def norm_value(self):
        return standardize(self.current_value, self.mean, self.std)
