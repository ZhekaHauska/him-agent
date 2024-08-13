#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from hima.common.scheduler import Scheduler
from hima.common.sdr import RateSdr, AnySparseSdr, OutputMode, unwrap_as_rate_sdr
from hima.common.sdr_array import SdrArray
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.se_dense import (
    oja_krotov_kuderov_update,
    oja_krotov_update
)
from hima.experiments.temporal_pooling.stp.se_utils import sample_weights, WeightsDistribution


class KrotovLayer:
    """
    A competitive network implementation from Krotov-Hopfield.
    Source: Unsupervised learning by competing hidden units
        https://pnas.org/doi/full/10.1073/pnas.1820458116
    """
    rng: Generator

    # input
    feedforward_sds: Sds

    # potentiation and learning
    learning_rate: float

    # output
    output_sds: Sds
    output_mode: OutputMode

    # connections
    weights: npt.NDArray[float]
    lebesgue_p: float

    def __init__(
            self, *, seed: int, feedforward_sds: Sds, output_sds: Sds,
            weights_distribution: str = 'normal', lebesgue_p: float,
            init_radius: float = 10.0, inhibitory_ratio: float = 0.5,
            repu_n: float, neg_hebb_delta: float,
            learning_rate: float, adaptive_lr: bool = False,
            lr_range: tuple[float, float] = (0.00001, 0.1), persistent_signs: bool = False,

            print_stats_schedule: int = 10_000,
            **kwargs
    ):
        print(f'kwargs: {kwargs}')
        self.rng = np.random.default_rng(seed)

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode.RATE

        weights_distribution = WeightsDistribution[weights_distribution.upper()]
        w_shape = (self.output_size, self.ff_size)
        self.lebesgue_p = lebesgue_p
        self.weights = sample_weights(
            self.rng, w_shape, weights_distribution, init_radius, self.lebesgue_p,
            inhibitory_ratio=inhibitory_ratio
        )
        self.radius = self.get_radius()

        self.match_p = lebesgue_p - 1
        self.repu_n = repu_n
        self.weights_pow_p = self.get_weight_pow_p()

        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        if self.adaptive_lr:
            self.lr_range = lr_range
            self.pos_log_radius = self.get_pos_log_radius()
        self.hebb_delta = np.array([1.0, -neg_hebb_delta])
        self.persistent_signs = persistent_signs

        self.cnt = 0
        self.loops = 0
        print(f'Init Krotov: {self.lebesgue_p}-norm | R = {self.avg_radius:.3f}')

        # # stats collection
        slow_lr = LearningRateParam(window=40_000)
        fast_lr = LearningRateParam(window=10_000)
        self.computation_speed = MeanValue(lr=slow_lr)
        self.slow_output_trace = MeanValue(
            size=self.output_size, lr=slow_lr, initial_value=self.output_sds.sparsity
        )
        self.fast_output_sdr_size_trace = MeanValue(
            lr=fast_lr, initial_value=self.output_sds.active_size
        )
        self.print_stats_scheduler = Scheduler(print_stats_schedule)

    def compute_batch(self, input_sdrs: SdrArray, learn: bool = False) -> SdrArray:
        output_sdr, run_time = self._compute_batch(input_sdrs, learn)
        # put average time per SDR
        self.computation_speed.put(run_time / len(input_sdrs))
        return output_sdr

    def compute(self, input_sdr: AnySparseSdr, learn: bool = False) -> AnySparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.computation_speed.put(run_time)
        return output_sdr

    @timed
    def _compute(self, input_sdr: AnySparseSdr, learn: bool) -> AnySparseSdr:
        sdr, value = unwrap_as_rate_sdr(input_sdr)
        x = np.zeros(self.ff_size)
        x[sdr] = value

        p, hb_delta = self.lebesgue_p, self.hebb_delta

        u = self.match_input(x)

        sdr = np.flatnonzero(u > 0)
        y = u[sdr] ** self.repu_n
        y /= y.sum()

        thr = 0.2 / len(sdr)
        alive_mask = y > thr
        sdr = sdr[alive_mask]
        y = y[alive_mask]

        output_sdr = RateSdr(sdr, y)
        if not learn or sdr.size == 0:
            return output_sdr

        kp1 = self.output_sds.active_size + 1
        if len(y) >= kp1:
            top_kp1_ix = sdr[np.argpartition(y, -kp1)[-kp1:]]
        else:
            top_kp1_ix = np.argpartition(u, -kp1)[-kp1:]

        # [best, worst] in top K+1
        ixs = np.array([
            top_kp1_ix[np.argmax(u[top_kp1_ix])],
            top_kp1_ix[np.argmin(u[top_kp1_ix])]
        ], dtype=int)

        lr = self.get_adaptive_lr(ixs) if self.adaptive_lr else self.learning_rate

        if self.persistent_signs:
            oja_krotov_kuderov_update(self.weights, ixs, x, u, hb_delta, lr)
        else:
            oja_krotov_update(self.weights, ixs, x, u, hb_delta, lr)

        self.weights_pow_p[ixs, :] = self.get_weight_pow_p(ixs)
        self.radius[ixs] = self.get_radius(ixs)
        if self.adaptive_lr:
            self.pos_log_radius[ixs] = self.get_pos_log_radius(ixs)

        # update winners activation stats
        self.fast_output_sdr_size_trace.put(len(output_sdr.sdr))

        self.cnt += 1

        if self.print_stats_scheduler.tick():
            self.print_stats(output_sdr.sdr, output_sdr.values, u)
        # if self.cnt % 10000 == 0:
        #     self.plot_weights_distr()
        # if self.cnt % 10000 == 0:
        #     self.plot_activation_distr(sdr, u, y)

        return output_sdr

    @timed
    def _compute_batch(self, input_sdrs: SdrArray, learn: bool) -> SdrArray:
        batch_size = len(input_sdrs)
        output_sdrs = []

        xs = input_sdrs.get_batch_dense(np.arange(batch_size))
        p, hb_delta = self.lebesgue_p, self.hebb_delta

        us = self.match_input(xs.T).T

        for i in range(batch_size):
            x, u = xs[i], us[i]

            sdr = np.flatnonzero(u > 0)
            y = u[sdr] ** self.repu_n
            y /= y.sum() + 1e-30

            thr = 0.2 / len(sdr)
            alive_mask = y > thr
            sdr = sdr[alive_mask]
            y = y[alive_mask]

            output_sdr = RateSdr(sdr, y)
            output_sdrs.append(output_sdr)

            if not learn or sdr.size == 0:
                continue

            kp1 = self.output_sds.active_size + 1
            if len(y) >= kp1:
                top_kp1_ix = sdr[np.argpartition(y, -kp1)[-kp1:]]
            else:
                top_kp1_ix = np.argpartition(u, -kp1)[-kp1:]

            # [best, worst] in top K+1
            ixs = np.array([
                top_kp1_ix[np.argmax(u[top_kp1_ix])],
                top_kp1_ix[np.argmin(u[top_kp1_ix])]
            ], dtype=int)

            lr = self.get_adaptive_lr(ixs) if self.adaptive_lr else self.learning_rate

            if self.persistent_signs:
                oja_krotov_kuderov_update(self.weights, ixs, x, u, hb_delta, lr)
            else:
                oja_krotov_update(self.weights, ixs, x, u, hb_delta, lr)

            self.weights_pow_p[ixs, :] = self.get_weight_pow_p(ixs)
            self.radius[ixs] = self.get_radius(ixs)
            if self.adaptive_lr:
                self.pos_log_radius[ixs] = self.get_pos_log_radius(ixs)

            # update winners activation stats
            self.fast_output_sdr_size_trace.put(len(output_sdr.sdr))
            self.slow_output_trace.put(output_sdr.values, output_sdr.sdr)
            self.cnt += 1

            # if self.cnt % 10000 == 0:
            #     self.plot_weights_distr()
            # if self.cnt % 10000 == 0:
            #     self.plot_activation_distr(sdr, u, y)

        if self.print_stats_scheduler.tick(batch_size):
            self.print_stats(output_sdrs[-1].sdr, output_sdrs[-1].values, us[-1])

        return SdrArray(sparse=output_sdrs, sdr_size=self.output_size)

    def print_stats(self, sdr, y, u):
        sorted_values = np.sort(y)
        ac_size = self.output_sds.active_size
        active_mass = sorted_values[-ac_size:].sum()
        biases = np.log(self.output_rate)
        print(
            f'{self.avg_radius:.3f} {self.output_active_size:.1f}'
            f'| {biases.mean():.2f} [{biases.min():.2f}; {biases.max():.2f}]'
            f'| {u.min():.3f}  {u.max():.3f}'
            f'| {self.weights.mean():.3f}: {self.weights.min():.3f}  {self.weights.max():.3f}'
            f'| {active_mass:.3f} {y.sum():.3f}  {sdr.size}'
        )

    def match_input(self, x):
        w, p = self.weights, self.match_p
        if p != 1.0:
            w = self.weights_pow_p
        return np.dot(w, x)

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.lebesgue_p
        w = self.weights if ixs is None else self.weights[ixs]
        return np.sum(np.abs(w) ** p, axis=-1) ** (1 / p)

    def get_weight_pow_p(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.match_p
        w = self.weights if ixs is None else self.weights[ixs]
        return np.sign(w) * (np.abs(w) ** p)

    def get_relative_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.clip(
            np.abs(np.log2(np.maximum(r, 0.001))),
            0.05, 4.0
        )

    def get_pos_log_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.log2(np.maximum(r, 1.0))

    def get_adaptive_lr(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        base_lr = self.learning_rate
        rs = self.pos_log_radius if ixs is None else self.pos_log_radius[ixs]
        return np.clip(base_lr * rs, *self.lr_range)

    @property
    def avg_radius(self):
        return self.radius.mean()

    def plot_activation_distr(self, sdr, u, y):
        k = self.output_sds.active_size
        ixs_ranked = sdr[np.argsort(y)][::-1]
        kth, eth = ixs_ranked[k - 1], ixs_ranked[-1]
        import matplotlib.pyplot as plt
        plt.hist(u, bins=50)
        plt.vlines([u[kth], u[eth]], 0, 20, colors=['r', 'y'])
        plt.show()
        _y = np.cumsum(np.sort(y))[::-1]
        _y /= _y[0]
        plt.plot(_y)
        plt.vlines(k, 0, 1, color='r')
        plt.show()

    def plot_weights_distr(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        w, r = self.weights, self.radius
        w_p, r_p = self.get_weight_pow_p(), self.radius ** 2
        w = w / np.expand_dims(r, -1)
        w_p = w_p / np.expand_dims(r_p, -1)
        sns.histplot(w.flatten())
        sns.histplot(w_p.flatten())
        plt.show()

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
        return self.fast_output_sdr_size_trace.get()

    @property
    def output_relative_rate(self):
        target_rate = self.output_sds.sparsity
        return self.output_rate / target_rate

    def output_entropy(self):
        return entropy(self.output_rate)


# @numba.jit(nopython=True, cache=True)
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
