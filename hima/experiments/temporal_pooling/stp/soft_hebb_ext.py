#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numba
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


class SoftHebbLayer:
    """
    A competitive SoftHebb network implementation.
    """
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
            init_radius: float, min_active_mass: float, min_mass: float, beta_lr: float,
            **kwargs
    ):
        print(f'kwargs: {kwargs}')

        self.rng = np.random.default_rng(seed)
        self.comp_name = None

        self.feedforward_sds = Sds.make(feedforward_sds)

        self.sparse_input = np.empty(0, dtype=int)
        self.dense_input = np.zeros(self.ff_size, dtype=float)
        self.is_empty_input = True

        self.output_sds = Sds.make(output_sds)

        self.potentials = np.zeros(self.output_size, dtype=float)
        self.learning_rate = learning_rate
        self.min_active_mass = min_active_mass
        self.min_mass = min_mass

        shape = (self.output_size, self.ff_size)
        req_radius = init_radius
        init_std = req_radius * np.sqrt(np.pi / 2 / self.ff_size)
        self.weights = self.rng.normal(loc=0.0, scale=init_std, size=shape)
        self.radius = self.get_radius()

        self.beta = 10.0
        self.beta_lr = beta_lr
        self.threshold = min(1 / self.output_sds.size, self.output_sds.active_size ** (-2))

        bias = np.log(1 / self.output_size) / self.beta
        self.biases = self.rng.normal(loc=bias, scale=0.001, size=self.output_size)
        self.cnt = 0
        self.loops = 0
        print(f'init_std: {init_std:.3f} | {self.avg_radius:.3f}')

        # # stats collection
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

        x, w, b, t = self.dense_input, self.weights, self.biases, self.threshold
        u = w @ x
        q = -np.log(self.avg_radius)
        y = softmax(u + q * b, beta=self.beta)

        loops, sdr, values, mass = get_important(y, t, self.min_mass)
        output_sdr = RateSdr(sdr, values / (mass + 1e-30))
        self.accept_output(output_sdr, learn=learn)

        if not learn:
            return output_sdr

        lr = self.learning_rate
        lr = lr * self.relative_radius + 0.0001

        _u = np.expand_dims(u, -1)
        _x = np.expand_dims(x, 0)
        _y = np.expand_dims(y, -1)
        _lr = np.expand_dims(lr, -1)

        d_weights = _y[sdr] * (_x - w[sdr] * _u[sdr])
        d_weights /= np.abs(d_weights).max() + 1e-30
        self.weights[sdr, :] += _lr[sdr] * d_weights
        self.radius[sdr] = self.get_radius(sdr)

        self.biases = np.log(self.output_rate)

        beta_lr = self.beta_lr
        sorted_values = np.sort(values)
        ac_size = self.output_sds.active_size
        active_mass = sorted_values[-ac_size:].sum()

        if len(sdr) < ac_size:
            d_beta = -0.02
        elif active_mass < self.min_active_mass or active_mass > self.min_mass:
            target_mass = (self.min_active_mass + self.min_mass) / 2
            rel_mass = np.clip(active_mass / target_mass, 0.7, 1.2)
            # less -> neg (neg log) -> increase beta and vice versa
            d_beta = -np.log(rel_mass)
        else:
            d_beta = 0.0

        self.beta *= np.exp(beta_lr * d_beta)
        self.beta = np.clip(self.beta, 1e-3, 1e+4)

        exp_missing_mass = (1.0 - self.min_mass) / 2
        rel_missing_mass = (1.0 - mass) / exp_missing_mass
        rel_missing_mass = np.clip(rel_missing_mass, 0.75, 1.1)
        d_thr = -loops**2 if loops > 1 else -0.5 * np.log(rel_missing_mass)
        self.threshold += 0.01 * beta_lr * d_thr

        self.loops += loops
        self.cnt += 1
        if self.cnt % 1000 == 0:
            low_y = y[y <= t]
            low_mx = 0. if low_y.size == 0 else low_y.max()
            print(
                f'{self.avg_radius:.3f} {self.output_entropy():.3f} {self.output_active_size:.1f}'
                f'| {self.beta:.1f} {self.threshold:.4f}'
                f'| {self.biases.mean():.3f} [{self.biases.min():.3f}; {self.biases.max():.3f}]'
                f'| {low_mx:.4f}  {y.max():.3f}'
                f'| {active_mass:.3f} {values.sum():.3f}  {sdr.size}'
                f'| {self.loops/self.cnt:.3f}'
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

    @property
    def output_relative_rate(self):
        target_rate = self.output_sds.sparsity
        return self.output_rate / target_rate

    def output_entropy(self):
        return entropy(self.output_rate)


@numba.jit(nopython=True, cache=True)
def get_important(a, min_val, min_mass):
    i = 1
    while True:
        sdr = np.flatnonzero(a > min_val)
        values = a[sdr]
        mass = values.sum()
        if mass >= min_mass:
            return i, sdr, values, mass
        min_val /= 4
        i += 1
