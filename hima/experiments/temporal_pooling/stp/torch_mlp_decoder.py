#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import torch
from torch import nn, optim

from hima.common.sdr import SparseSdr, DenseSdr
from hima.common.sdrr import RateSdr, AnySparseSdr, OutputMode, split_sdr_values
from hima.common.sds import Sds


class MlpDecoder:
    feedforward_sds: Sds
    output_sds: Sds
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
            learn_on_sdr: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds
        self.output_mode = OutputMode[output_mode.upper()]

        shape = (output_sds.size, feedforward_sds.size)

        self.lr = learning_rate
        self.sdr_predictor = nn.Sequential(
            nn.Linear(shape[0], shape[1], dtype=float),
            nn.Sigmoid()
        )
        self.values_predictor = nn.Linear(shape[0], shape[1], dtype=float)
        self.optim = optim.Adam(
            itertools.chain(
                self.sdr_predictor.parameters(),
                self.values_predictor.parameters()
            ),
            lr=self.lr
        )
        self.loss = 0.

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
        x = torch.from_numpy(self.dense_input)
        sdr = self.sdr_predictor(x)
        values = sdr
        # values *= self.values_predictor(x)
        return values

    def learn(
            self, input_sdr: AnySparseSdr, gt_sdr: AnySparseSdr,
            prediction: AnySparseSdr | DenseSdr
    ):
        self.n_updates += 1
        self.accept_ground_truth(gt_sdr)

        if prediction is None:
            self.accept_input(input_sdr)
            prediction = self.predict()

        target = torch.from_numpy(self.dense_gt)

        self.optim.zero_grad()
        loss_func = nn.MSELoss()
        loss = loss_func(prediction, target)
        loss.backward()
        self.optim.step()

        if self.collect_errors:
            self.errors.append(loss.item())

    def to_sdr(self, prediction: torch.Tensor) -> AnySparseSdr:
        prediction = prediction.detach().numpy()

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
