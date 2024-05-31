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
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.se import get_normal_std


class SoftHebbLayer:
    """A competitive SoftHebb network implementation. Near-exact implementation."""
    rng: Generator

    # input
    feedforward_sds: Sds
    # input cache
    sparse_input: SparseSdr
    dense_input: DenseSdr

    # potentiation and learning
    learning_rate: float

    # output
    output_sds: Sds
    output_mode: OutputMode

    # connections
    weights: npt.NDArray[float]

    lebesgue_p: float

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
        self.output_mode = OutputMode.RATE

        self.lebesgue_p = 2.0
        self.learning_rate = learning_rate

        shape = (self.output_size, self.ff_size)
        init_std = get_normal_std(init_radius, self.ff_size, self.lebesgue_p)
        self.weights = self.rng.normal(loc=0.0, scale=init_std, size=shape)
        self.radius = self.get_radius()

        self.beta = beta
        self.threshold = threshold
        self.cnt = 0
        print(f'init_std: {init_std:.3f} | {self.avg_radius:.3f}')

        slow_lr = LearningRateParam(window=40_000)
        fast_lr = LearningRateParam(window=10_000)
        self.computation_speed = MeanValue(lr=slow_lr)
        self.slow_output_trace = MeanValue(
            size=self.output_size, lr=slow_lr, initial_value=self.output_sds.sparsity
        )
        self.slow_output_sdr_size_trace = MeanValue(
            lr=fast_lr, initial_value=self.output_sds.active_size
        )

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
        output_sdr = RateSdr(sdr, values / (values.sum() + 1e-30))
        self.accept_output(output_sdr, learn=learn)

        if not learn or sdr.size == 0:
            return output_sdr

        _lr = lr * self.relative_radius
        _lr = np.expand_dims(_lr, -1)
        _u = np.expand_dims(u, -1)
        _x = np.expand_dims(x, 0)
        _y = np.expand_dims(y, -1)

        d_weights = _y[sdr] * (_x - w[sdr] * _u[sdr])
        d_weights /= np.abs(d_weights).max() + 1e-30
        self.weights[sdr, :] += _lr[sdr] * d_weights
        self.radius[sdr] = self.get_radius(sdr)

        self.cnt += 1
        if self.cnt % 10000 == 0:
            print(
                f'{self.avg_radius:.5f}  {self.output_entropy():.3f} {self.output_active_size:.1f}'
                f'| {self.weights.mean():.3f}: {self.weights.min():.3f}  {self.weights.max():.3f}'
                f'| {y[y<=thr].max():.3f}  {values.max():.3f}'
                f'| {values.sum():.3f}  {values.size}'
            )

        return output_sdr

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        if ixs is None:
            return np.sqrt(np.sum(self.weights ** 2, axis=-1))
        return np.sqrt(np.sum(self.weights[ixs] ** 2, axis=-1))

    @property
    def relative_radius(self):
        return np.abs(np.log2(np.maximum(self.radius, 0.001)))

    @property
    def avg_radius(self):
        return self.radius.mean()

    def accept_input(self, sdr: AnySparseSdr, *, learn: bool):
        """Accept new input and move to the next time step"""
        sdr, value = split_sdr_values(sdr)

        # forget prev SDR
        self.dense_input[self.sparse_input] = 0.

        # set new SDR
        self.sparse_input = sdr
        self.dense_input[self.sparse_input] = value

    def accept_output(self, sdr: AnySparseSdr, *, learn: bool):
        sdr, value = split_sdr_values(sdr)

        if not learn or sdr.shape[0] == 0:
            return

        # update winners activation stats
        self.slow_output_trace.put(value, sdr)
        self.slow_output_sdr_size_trace.put(len(sdr))

    @property
    def ff_size(self):
        return self.feedforward_sds.size

    @property
    def output_size(self):
        return self.output_sds.size

    @property
    def output_rate(self):
        return self.slow_output_trace.get()

    @property
    def output_active_size(self):
        return self.slow_output_sdr_size_trace.get()

    def output_entropy(self):
        return entropy(self.output_rate)
