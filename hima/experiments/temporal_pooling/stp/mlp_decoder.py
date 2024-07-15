#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hima.common.sdr import SparseSdr, DenseSdr, RateSdr, AnySparseSdr, OutputMode, split_sdr_values
from hima.common.sds import Sds


class MlpDecoder:
    feedforward_sds: Sds
    output_sds: Sds
    use_sigmoid: bool
    output_mode: OutputMode

    # cache
    sparse_input: SparseSdr
    dense_input: npt.NDArray[float]
    sparse_gt: SparseSdr
    dense_gt: npt.NDArray[float]

    # learning stage tracking
    n_updates: int

    def __init__(
            self,
            feedforward_sds: Sds, output_sds: Sds,
            seed: int = None,
            weights_scale: float = 0.01,
            learning_rate: float = 0.25, power_t: float = 0.05,
            total_updates_required: int = 100_000, epoch_size: int = 4_000,
            collect_errors: bool = False,
            output_mode: str = 'binary',
            use_sigmoid: bool = False
    ):
        self.rng = np.random.default_rng(seed)
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds
        self.use_sigmoid = use_sigmoid
        self.output_mode = OutputMode[output_mode.upper()]

        shape = (output_sds.size, feedforward_sds.size)

        self.weights = self.rng.normal(0., weights_scale, size=shape)
        self.lr = learning_rate
        self.power_t = power_t

        self.n_updates = 0
        self.total_updates_required = total_updates_required
        self.epoch_size = epoch_size

        self.sparse_input = []
        self.dense_input = np.zeros(self.feedforward_sds.size, dtype=float)
        self.sparse_gt = []
        self.dense_gt = np.zeros(self.output_sds.size, dtype=float)

        self.collect_errors = collect_errors
        if self.collect_errors:
            from collections import deque
            self.errors = deque(maxlen=100)

    def decode(self, input_sdr):
        self.accept_input(input_sdr)
        return self.predict()

    def predict(self):
        values = self.weights @ self.dense_input
        if self.use_sigmoid:
            values = 1. / (1 + np.exp(-values))
        return values

    def learn(
            self, input_sdr: AnySparseSdr, gt_sdr: AnySparseSdr,
            prediction: AnySparseSdr | DenseSdr
    ):
        # if self.n_updates >= self.total_updates_required:
        #     return

        # every stage update will be approximately every `stage`-th time
        epoch = (1 + self.n_updates // self.epoch_size) ** 0.7
        # if self.rng.random() >= 1. / epoch:
        #     return

        self.n_updates += 1
        self.accept_input(input_sdr)
        self.accept_ground_truth(gt_sdr)

        if prediction is None:
            prediction = self.predict()

        full_k = 1.
        sum_k = 0.
        sdr_k = 0.
        k_norm = sum_k + full_k + sdr_k
        sum_k, full_k, sdr_k = sum_k / k_norm, full_k / k_norm, sdr_k / k_norm

        lr = self.lr / (epoch ** self.power_t)

        loss_derivative = None
        if full_k > 0.:
            loss_derivative = prediction - self.dense_gt
            self.weights -= np.outer(loss_derivative, full_k * lr * self.dense_input)

        if sum_k > 0.:
            sum_loss_derivative = prediction.sum() - self.dense_gt.sum()
            self.weights -= np.expand_dims(sum_k * lr * sum_loss_derivative * self.dense_input, axis=0)

        if self.collect_errors and loss_derivative is not None:
            self.errors.append(np.abs(loss_derivative).mean())

    def to_sdr(self, prediction: DenseSdr) -> AnySparseSdr:
        n_winners = self.output_sds.active_size
        winners = np.argpartition(prediction, -n_winners)[-n_winners:]
        winners.sort()
        winners = winners[prediction[winners] > 0]

        if self.output_mode == OutputMode.RATE:
            values = np.clip(prediction[winners], 0., 1.)
            return RateSdr(winners, values=values)
        return winners

    def accept_input(self, sdr: AnySparseSdr):
        """Accept new input and move to the next time step"""
        sdr, values = split_sdr_values(sdr)

        # forget prev SDR
        self.dense_input[self.sparse_input] = 0.

        # set new SDR
        self.sparse_input = sdr
        self.dense_input[self.sparse_input] = values

    def accept_ground_truth(self, sdr: AnySparseSdr):
        """Accept new input and move to the next time step"""
        sdr, values = split_sdr_values(sdr)

        # forget prev SDR
        self.dense_gt[self.sparse_gt] = 0.

        # set new SDR
        self.sparse_gt = sdr
        self.dense_gt[self.sparse_gt] = values
