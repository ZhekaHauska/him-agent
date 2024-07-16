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

from hima.common.sdr import SparseSdr, DenseSdr, RateSdr, AnySparseSdr, OutputMode, split_sdr_values
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.common.utils import softmax
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.se import get_normal_std


class SoftHebbLayer:
    """
    A competitive SoftHebb network implementation with several modifications:
        - adaptive softmax beta. Such that the K most active neurons (=output active size)
            should make ~[minM, maxM] of the total mass. While the actual active size
            will be ~2-3 times larger, however it still indirectly/softly defines sparsity.
        - adaptive threshold. It is adjusted to keep the total output mass close to 1.
        - adaptive learning rate. Since learning rule forces neuron's weights norm to 1,
            LR decays to 0 as the norm approaches 1. TODO: set clipping rule as in KrotovExt impl.
        - TODO: remove unnormalized delta weights.

    """
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
            init_radius: float, min_active_mass: float, min_mass: float, beta_lr: float,
            **kwargs
    ):
        print(f'kwargs: {kwargs}')
        self.rng = np.random.default_rng(seed)

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.sparse_input = np.empty(0, dtype=int)
        self.dense_input = np.zeros(self.ff_size, dtype=float)

        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode.RATE

        self.lebesgue_p = 2.0
        self.learning_rate = learning_rate
        self.min_active_mass = min_active_mass
        self.min_mass = min_mass

        shape = (self.output_size, self.ff_size)
        init_std = get_normal_std(init_radius, self.ff_size, self.lebesgue_p)
        self.weights = self.rng.normal(loc=0.001, scale=init_std, size=shape)
        self.radius = self.get_radius()
        self.relative_radius = self.get_relative_radius()

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

        x, w, b, thr = self.dense_input, self.weights, self.biases, self.threshold
        u = w @ x
        q = -self.relative_radius
        y = softmax(u + q * b, beta=self.beta)

        sdr, values, mass = get_important(y, thr)
        output_sdr = RateSdr(sdr, values / (mass + 1e-30))
        self.accept_output(output_sdr, learn=learn)

        if not learn or sdr.size == 0:
            return output_sdr

        lr = self.learning_rate
        lr = lr * self.relative_radius[sdr]

        _u = np.expand_dims(u, -1)
        _x = np.expand_dims(x, 0)
        _y = np.expand_dims(y, -1)
        _lr = np.expand_dims(lr, -1)

        d_weights = _y[sdr] * (_x - w[sdr] * _u[sdr])
        d_weights /= np.abs(d_weights).max() + 1e-30
        self.weights[sdr, :] += _lr * d_weights
        self.radius[sdr] = self.get_radius(sdr)
        self.relative_radius[sdr] = self.get_relative_radius(sdr)

        self.biases = np.log(self.output_rate)

        beta_lr = self.beta_lr
        top_k = min(self.output_sds.active_size, len(values))
        active_mass = values[np.argpartition(values, -top_k)[-top_k:]].sum()

        if len(sdr) < top_k:
            d_beta = -0.02
        elif active_mass < self.min_active_mass or active_mass > self.min_mass:
            target_mass = (self.min_active_mass + self.min_mass) / 2
            rel_mass = max(0.1, active_mass / target_mass)
            # less -> neg (neg log) -> increase beta and vice versa
            d_beta = -np.log(rel_mass)
        else:
            d_beta = 0.0

        self.beta *= np.exp(beta_lr * np.clip(d_beta, -1.0, 1.0))
        self.beta += beta_lr * d_beta
        self.beta = np.clip(self.beta, 1e-3, 1e+4)

        exp_missing_mass = (1.0 - self.min_mass) / 2
        rel_missing_mass = (1.0 - mass) / exp_missing_mass
        rel_missing_mass = np.clip(rel_missing_mass, 0.75, 1.1)
        d_thr = -0.5 * np.log(rel_missing_mass)
        self.threshold += 0.01 * beta_lr * d_thr

        self.loops += 1
        self.cnt += 1
        if self.cnt % 2000 == 0:
            low_y = y[y <= thr]
            low_mx = 0. if low_y.size == 0 else low_y.max()
            print(
                f'{self.avg_radius:.3f} {self.output_entropy():.3f} {self.output_active_size:.1f}'
                f'| {self.beta:.1f} {self.threshold:.4f}'
                f'| {self.biases.mean():.3f} [{self.biases.min():.3f}; {self.biases.max():.3f}]'
                f'| {self.weights.mean():.3f}: {self.weights.min():.3f}  {self.weights.max():.3f}'
                f'| {low_mx:.4f}  {y.max():.3f}'
                f'| {active_mass:.3f} {values.sum():.3f}  {sdr.size}'
                f'| {self.loops/self.cnt:.3f}'
            )
        # if self.cnt % 10000 == 0:
        #     import seaborn as sns
        #     from matplotlib import pyplot as plt
        #     w, r = self.weights, self.radius
        #     w_p, r_p = self.get_weight_pow_p(p=2), self.radius ** 2
        #     w = w / np.expand_dims(r, -1)
        #     w_p = w_p / np.expand_dims(r_p, -1)
        #     sns.histplot(w.flatten())
        #     sns.histplot(w_p.flatten())
        #     plt.show()

        return output_sdr

    def get_weight_pow_p(self, p: float, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        w = self.weights if ixs is None else self.weights[ixs]
        return np.sign(w) * (np.abs(w) ** p)

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        if ixs is None:
            w = self.weights
            return np.sqrt(np.sum(w ** 2, axis=-1))

        w = self.weights[ixs]
        return np.sqrt(np.sum(w ** 2, axis=-1))

    def get_relative_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.clip(
            np.abs(np.log2(np.maximum(r, 0.001))),
            0.05, 4.0
        )

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

    @property
    def output_relative_rate(self):
        target_rate = self.output_sds.sparsity
        return self.output_rate / target_rate

    def output_entropy(self):
        return entropy(self.output_rate)


@numba.jit(nopython=True, cache=True)
def get_important(a, min_val):
    sdr = np.flatnonzero(a > min_val)
    values = a[sdr]
    mass = values.sum()
    return sdr, values, mass
