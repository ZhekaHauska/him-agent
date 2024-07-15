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

from hima.common.sdr import SparseSdr, DenseSdr, RateSdr, AnySparseSdr, OutputMode, split_sdr_values
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.modules.baselines.lstm import to_numpy, symexp


class SymExpModule(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return symexp(x)


class MlpDecoder:
    feedforward_sds: Sds
    output_sds: Sds
    output_mode: OutputMode

    # cache
    sparse_input: SparseSdr
    dense_input: npt.NDArray[float]
    sparse_gt: SparseSdr
    dense_gt_values: npt.NDArray[float]
    dense_gt_sdr: npt.NDArray[float]

    # learning stage tracking
    n_updates: int

    accumulated_loss: float | torch.Tensor
    accumulated_loss_steps: int | None
    loss_propagation_schedule: int

    def __init__(
            self,
            feedforward_sds: Sds, output_sds: Sds,
            seed: int = None,
            weights_scale: float = 0.01,
            learning_rate: float = 0.25, power_t: float = 0.05,
            total_updates_required: int = 100_000, epoch_size: int = 4_000,
            collect_errors: bool = False,
            output_mode: str = 'binary',
            loss_propagation_schedule: int = 5,
    ):
        self.rng = np.random.default_rng(seed)
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds
        self.output_mode = OutputMode[output_mode.upper()]

        shape = (feedforward_sds.size, output_sds.size)

        self.lr = learning_rate
        self.sdr_predictor = nn.Sequential(
            nn.Linear(shape[0], shape[1], dtype=float),
            SymExpModule(),
            nn.Sigmoid()
        )
        self.values_predictor = nn.Sequential(
            nn.Linear(shape[1], shape[1], dtype=float),
        )
        self.optim = optim.Adam(
            itertools.chain(
                self.sdr_predictor.parameters(),
                self.values_predictor.parameters()
            ),
            lr=self.lr
        )
        self.accumulated_loss = 0.
        self.accumulated_loss_steps = 0
        self.loss_propagation_schedule = loss_propagation_schedule

        self.n_updates = 0
        self.total_updates_required = total_updates_required
        self.epoch_size = epoch_size

        self.sparse_input = []
        self.dense_input = np.zeros(self.feedforward_sds.size, dtype=float)
        self.sparse_gt = []
        self.dense_gt_values = np.zeros(self.output_sds.size, dtype=float)
        self.dense_gt_sdr = np.zeros(self.output_sds.size, dtype=float)

        self.collect_errors = collect_errors
        if self.collect_errors:
            from collections import deque
            self.errors = deque(maxlen=100)

    def decode(self, input_sdr):
        self.accept_input(input_sdr)
        return self.predict()

    def predict(self):
        x = torch.from_numpy(self.dense_input)
        sdr_probs = self.sdr_predictor(x)

        # sampler = torch.distributions.Bernoulli(probs=sdr_probs)
        # sdr = sampler.sample()

        # sampler = torch.distributions.RelaxedBernoulli(torch.tensor([1.0]), probs=sdr_probs)
        # sdr = sampler.rsample()

        # values = sdr
        # values = sdr * self.values_predictor(sdr_probs.detach())
        # return sdr_probs, values

        values = self.values_predictor(sdr_probs.detach())
        return sdr_probs, values

    def learn(
            self, input_sdr: AnySparseSdr, gt_sdr: AnySparseSdr,
            prediction: AnySparseSdr | DenseSdr
    ):
        self.n_updates += 1
        self.accept_ground_truth(gt_sdr)

        if prediction is None:
            self.accept_input(input_sdr)
            prediction = self.predict()

        target_values = torch.from_numpy(self.dense_gt_values)
        target_sdr = torch.from_numpy(self.dense_gt_sdr)

        self.optim.zero_grad()
        sdr_loss_func = nn.BCELoss()
        value_loss_func = nn.MSELoss()

        sdr_probs, values = prediction

        loss = value_loss_func(values, target_values) + sdr_loss_func(sdr_probs, target_sdr)

        self.accumulated_loss += loss
        self.accumulated_loss_steps += 1
        self.backpropagate_loss()

        if self.collect_errors:
            self.errors.append(loss.item())

    def backpropagate_loss(self):
        if self.accumulated_loss_steps <= self.loss_propagation_schedule:
            return

        self.optim.zero_grad()
        mean_loss = self.accumulated_loss / self.accumulated_loss_steps
        mean_loss.backward()
        self.optim.step()

        self.accumulated_loss = 0.
        self.accumulated_loss_steps = 0

    def to_sdr(self, prediction: torch.Tensor) -> AnySparseSdr:
        sdr, values = prediction
        sdr = to_numpy(sdr)
        values = to_numpy(values)

        n_winners = self.output_sds.active_size

        # NB: sample winners by the probability
        # NB2: add noise to simplify the case when there are no winners or less than needed
        # sdr = sdr + 1e-4
        # winners = self.rng.choice(
        #     self.output_sds.size, size=n_winners, p=sdr/sdr.sum(), replace=False
        # )
        # winners.sort()

        # NB: select winners by the constant threshold
        # winners = sdr > 0.5

        # NB: select winners with the highest probability
        winners = np.argpartition(sdr, -n_winners)[-n_winners:]
        winners.sort()
        winners = winners[sdr[winners] > 0]

        if self.output_mode == OutputMode.RATE:
            values = np.clip(values[winners], 0., 1.)
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
        self.dense_gt_values[self.sparse_gt] = 0.
        self.dense_gt_sdr[self.sparse_gt] = 0.

        # set new SDR
        self.sparse_gt = sdr
        self.dense_gt_values[self.sparse_gt] = values
        self.dense_gt_sdr[self.sparse_gt] = 1.
