#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

class AnomalyBasedFallingAsleep:
    anomaly_threshold: float
    alpha: float
    beta: float
    max_prob: float

    def __init__(
            self, anomaly_threshold: float, alpha: float,
            beta: float, max_prob: float,
    ):
        self.anomaly_threshold = anomaly_threshold
        self.alpha = alpha
        self.beta = beta
        self.max_prob = max_prob
