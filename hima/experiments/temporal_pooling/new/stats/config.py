#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.


class StatsMetricsConfig:
    mae_normalization: str
    symmetrical_similarity: bool

    distribution_metrics: str
    online_similarity_decay: float
    pmf_decay: float

    loss_normalization: bool
    loss_layer_discount: float

    def __init__(
            self, mae_normalization: str, symmetrical_similarity: bool,
            distribution_metrics: str,
            online_similarity_decay: float, pmf_decay: float,
            loss_normalization: bool, loss_layer_discount: float,
    ):
        self.mae_normalization = mae_normalization
        self.symmetrical_similarity = symmetrical_similarity
        self.distribution_metrics = distribution_metrics
        self.online_similarity_decay = online_similarity_decay
        self.pmf_decay = pmf_decay
        self.loss_normalization = loss_normalization
        self.loss_layer_discount = loss_layer_discount
