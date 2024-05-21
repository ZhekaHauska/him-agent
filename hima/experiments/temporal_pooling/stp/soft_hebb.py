#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from hima.common.sdr import SparseSdr, DenseSdr
from hima.common.sdrr import RateSdr, AnySparseSdr, OutputMode, split_sdr_values
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.common.utils import softmax
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam


class SoftHebbLayer:
    """A competitive SoftHebb network implementation."""
    rng: Generator

    # input
    feedforward_sds: Sds
    # input cache
    sparse_input: SparseSdr
    dense_input: DenseSdr

    # potentiation and learning
    potentials: npt.NDArray[float]
    learning_rate: float

    # output
    output_sds: Sds
    output_mode: OutputMode

    # connections
    weights: npt.NDArray[float]

    def __init__(
            self, *, seed: int, feedforward_sds: Sds, output_sds: Sds, learning_rate: float,
            init_radius: float, beta: float, threshold: float,
            **kwargs
    ):
        print(f'kwargs: {kwargs}')

        self.rng = np.random.default_rng(seed)
        self.comp_name = None

        self.feedforward_sds = Sds.make(feedforward_sds)

        self.sparse_input = np.empty(0, dtype=int)
        # use float not only to generalize to Rate SDR, but also to eliminate
        # inevitable int-to-float converting when we multiply it by weights
        self.dense_input = np.zeros(self.ff_size, dtype=float)
        self.is_empty_input = True

        self.output_sds = Sds.make(output_sds)

        self.potentials = np.zeros(self.output_size, dtype=float)
        self.learning_rate = learning_rate

        shape = (self.output_size, self.ff_size)
        req_radius = init_radius
        init_std = req_radius * np.sqrt(np.pi / 2 / self.ff_size)
        self.weights = self.rng.normal(loc=0.0, scale=init_std, size=shape)

        self.beta = beta
        self.threshold = threshold
        self.cnt = 0
        print(f'init_std: {init_std:.3f} | {self.avg_weight_norm:.3f}')

        slow_lr = LearningRateParam(window=400_000)
        self.computation_speed = MeanValue(lr=slow_lr)

    def compute(self, input_sdr: AnySparseSdr, learn: bool = False) -> AnySparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.computation_speed.put(run_time)
        return output_sdr

    @timed
    def _compute(self, input_sdr: AnySparseSdr, learn: bool) -> AnySparseSdr:
        self.accept_input(input_sdr, learn=learn)

        lr = self.learning_rate
        x, w = self.dense_input, self.weights
        u = w @ x
        y = softmax(u, beta=self.beta)

        # Fixed threshold
        thr = self.threshold
        sdr = np.flatnonzero(y > thr)
        values = y[sdr]
        v_sum = values.sum()
        if v_sum > 1e-12:
            values /= v_sum
        output_sdr = RateSdr(sdr, values)

        if not learn:
            return output_sdr

        _u = np.expand_dims(u, -1)
        _x = np.expand_dims(x, 0)
        _y = np.expand_dims(y, -1)

        r = np.sqrt(np.sum(w ** 2, axis=-1))
        _lr = lr * np.abs(np.log2(np.maximum(r, 0.001)))
        _lr = np.expand_dims(_lr, -1)
        d_weights = _y[sdr] * (_x - w[sdr] * _u[sdr])
        d_weights /= np.abs(d_weights).max() + 1e-30
        self.weights[sdr, :] += _lr[sdr] * d_weights

        self.cnt += 1
        if self.cnt % 1000 == 0:
            print(
                f'{self.avg_weight_norm:.5f}'
                f'| {y[y<=thr].max():.3f}  {y.max():.3f} |  {y[sdr].sum():.3f}    {y[sdr].size}'
            )

        return output_sdr

    @property
    def avg_weight_norm(self):
        return np.sqrt((self.weights ** 2).sum(-1)).mean()

    def accept_input(self, sdr: AnySparseSdr, *, learn: bool):
        """Accept new input and move to the next time step"""
        sdr, value = split_sdr_values(sdr)
        self.is_empty_input = len(sdr) == 0

        if not self.is_empty_input:
            l2_value = np.sqrt(np.sum(value**2))
            if l2_value > 1e-12:
                value /= l2_value

        # forget prev SDR
        self.dense_input[self.sparse_input] = 0.
        # reset potential
        self.potentials.fill(0.)

        # set new SDR
        self.sparse_input = sdr
        self.dense_input[self.sparse_input] = value

    @property
    def ff_size(self):
        return self.feedforward_sds.size

    @property
    def output_size(self):
        return self.output_sds.size

    @property
    def is_newborn_phase(self):
        return False
