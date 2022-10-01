#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.


class StatsMetricsConfig:
    mae_normalization: str

    prefix_similarity_discount: float
    loss_on_mae: bool
    loss_layer_discount: float

    symmetrical_similarity: bool

    def __init__(
            self, mae_normalization: str, prefix_similarity_discount: float,
            loss_layer_discount: float, loss_on_mae: bool, symmetrical_similarity: bool
    ):
        self.mae_normalization = mae_normalization
        self.prefix_similarity_discount = prefix_similarity_discount
        self.loss_layer_discount = loss_layer_discount
        self.loss_on_mae = loss_on_mae
        self.symmetrical_similarity = symmetrical_similarity
