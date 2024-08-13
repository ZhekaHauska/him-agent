#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from enum import Enum, auto

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from hima.common.scheduler import Scheduler
from hima.common.sdr import (
    RateSdr, AnySparseSdr, OutputMode,
    unwrap_as_rate_sdr
)
from hima.common.sdr_array import SdrArray
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.common.utils import softmax
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.se_utils import (
    WeightsDistribution,
    sample_weights
)


class NegativeHebbian(Enum):
    NO = 0
    RATE = auto()
    TOP_K = auto()


class FilterOutput(Enum):
    SOFT = 1
    HARD = auto()


class NormalizeOutput(Enum):
    NO = 0
    YES = auto()


class SoftHebbLayer:
    """A competitive SoftHebb network implementation. Near-exact implementation."""
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
            learning_rate: float,
            weights_distribution: str = 'normal', init_radius: float = 20.0,
            inhibitory_ratio: float = 0.5,
            adaptive_lr: bool = False, lr_range: tuple[float, float] = (0.00001, 0.1),
            normalize_dw: bool = False, persistent_signs: bool = False,
            # boosting via negative bias
            bias_boosting: bool = True,
            # activation threshold and softmax beta
            threshold: float = 0.001, output_extra: float = 0.5,
            beta: float = 10.0, beta_lr: float = 0.01,
            min_active_mass: float = None, min_mass: float = None,
            # others
            negative_hebbian: str = 'no', filter_output: str = 'soft',
            normalize_output: str = 'no',

            print_stats_schedule: int = 10_000,
            **kwargs
    ):
        print(f'kwargs: {kwargs}')
        self.rng = np.random.default_rng(seed)

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode.RATE

        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        if self.adaptive_lr:
            self.lr_range = lr_range
        self.persistent_signs = persistent_signs

        self.negative_hebbian = NegativeHebbian[negative_hebbian.upper()]
        self.filter_output = FilterOutput[filter_output.upper()]
        self.normalize_output = NormalizeOutput[normalize_output.upper()]

        self.lebesgue_p = 2.0
        weights_distribution = WeightsDistribution[weights_distribution.upper()]
        w_shape = (self.output_size, self.ff_size)
        self.weights = sample_weights(
            self.rng, w_shape, weights_distribution, init_radius, self.lebesgue_p,
            inhibitory_ratio=inhibitory_ratio
        )
        self.normalize_dw = normalize_dw
        self.radius = self.get_radius()
        self.relative_radius = self.get_relative_radius()

        self.bias_boosting = bias_boosting
        if self.bias_boosting:
            bias = np.log(1 / self.output_size)
            self.base_bias = bias
            self.biases = self.rng.normal(loc=bias, scale=0.001, size=self.output_size)

        self.threshold = threshold
        self.beta = beta
        self.adaptive_beta = beta_lr > 0.0
        if self.adaptive_beta:
            self.beta_lr = beta_lr
            self.threshold_lr = beta_lr / 100.0
            self.threshold = min(1 / self.output_sds.size, self.output_sds.active_size ** (-2))
            self.output_extra = output_extra

        self.min_active_mass = min_active_mass
        self.min_mass = min_mass

        self.cnt = 0
        print(f'Init SoftHebb: {self.lebesgue_p}-norm | R = {self.avg_radius:.3f}')

        slow_lr = LearningRateParam(window=40_000)
        fast_lr = LearningRateParam(window=10_000)
        self.computation_speed = MeanValue(lr=slow_lr)
        self.slow_output_trace = MeanValue(
            size=self.output_size, lr=slow_lr, initial_value=self.output_sds.sparsity
        )
        self.fast_output_sdr_size_trace = MeanValue(
            lr=fast_lr, initial_value=self.output_sds.active_size
        )
        if self.adaptive_beta:
            self.fast_active_mass_trace = MeanValue(lr=fast_lr, initial_value=self.min_active_mass)
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

        w = self.weights
        u_raw = np.dot(w, x)
        u = u_raw
        if self.bias_boosting:
            u = self.boost_potentials(u_raw)

        y = softmax(u, beta=self.beta)

        # Fixed threshold
        thr = self.threshold
        sdr = np.flatnonzero(y > thr)
        values = y[sdr]
        mass = values.sum()

        self.fast_output_sdr_size_trace.put(len(sdr))
        if self.filter_output == FilterOutput.HARD:
            o_sdr = sdr
        else:
            o_sdr = np.flatnonzero(y > 1e-4)

        o_values = y[o_sdr].copy()
        if self.normalize_output == NormalizeOutput.YES:
            o_values /= mass + 1e-30

        output_sdr = RateSdr(o_sdr, o_values)
        if not learn or sdr.size == 0:
            return output_sdr

        k = self.output_sds.active_size
        top_k = None
        if self.adaptive_beta or self.negative_hebbian == NegativeHebbian.TOP_K:
            cur_k = min(k, len(sdr))
            top_k = values[np.argpartition(values, -cur_k)[-cur_k:]]

        y = y[sdr]
        if self.negative_hebbian == NegativeHebbian.RATE:
            y = y - self.output_rate[sdr]
        elif self.negative_hebbian == NegativeHebbian.TOP_K:
            y = y - np.min(top_k)
            y[y < 0] *= 0.2

        lr = self.learning_rate
        if self.adaptive_lr:
            lr = self.get_adaptive_lr(sdr)

        _u = np.expand_dims(u_raw[sdr], -1)
        _x = np.expand_dims(x, 0)
        _y = np.expand_dims(y, -1)
        lr = np.expand_dims(lr, -1)

        d_weights = _y * (_x - w[sdr] * _u)
        if self.normalize_dw:
            d_weights /= np.abs(d_weights).max() + 1e-30
        self.weights[sdr, :] += lr * d_weights

        self.radius[sdr] = self.get_radius(sdr)
        self.relative_radius[sdr] = self.get_relative_radius(sdr)
        self.slow_output_trace.put(output_sdr.values, output_sdr.sdr)

        if self.bias_boosting:
            self.biases = np.log(self.output_rate)

        if self.adaptive_beta:
            beta_lr = self.beta_lr * np.sqrt(np.mean(lr))
            active_mass = top_k.sum()

            self.fast_active_mass_trace.put(active_mass)
            avg_active_mass = self.fast_active_mass_trace.get()
            avg_active_size = self.output_active_size

            d_beta = 0.0
            if avg_active_size < k:
                d_beta = -0.02
            elif avg_active_mass < self.min_active_mass or avg_active_mass > self.min_mass:
                target_mass = (self.min_active_mass + self.min_mass) / 2
                rel_mass = max(0.1, avg_active_mass / target_mass)
                # less -> neg (neg log) -> increase beta and vice versa
                d_beta = -np.log(rel_mass)

            if d_beta != 0.0:
                self.beta *= np.exp(beta_lr * np.clip(d_beta, -1.0, 1.0))
                self.beta += beta_lr * d_beta
                self.beta = np.clip(self.beta, 1e-3, 1e+4)

            thr_lr = self.threshold_lr * np.sqrt(np.mean(lr))
            if avg_active_size < k:
                d_thr = -1.0
            elif avg_active_size > k * (1.0 + 2 * self.output_extra):
                d_thr = 1.0
            else:
                d_thr = 0.1 * np.sign(avg_active_size - k * (1.0 + self.output_extra))

            self.threshold += thr_lr * d_thr
            self.threshold = max(1e-5, self.threshold)

        self.cnt += 1
        if self.print_stats_scheduler.tick():
            low_y = y[y <= thr]
            low_mx = 0. if low_y.size == 0 else low_y.max()
            stats = (
                f'{self.avg_radius:.3f} {self.output_entropy():.3f} {self.output_active_size:.1f}'
                f'| {self.beta:.1f} {self.threshold:.4f}'
            )
            if self.bias_boosting:
                stats += (
                    f'| {self.biases.mean():.3f} [{self.biases.min():.3f}; {self.biases.max():.3f}]'
                )
            stats += (
                f'| {self.weights.mean():.3f}: [{self.weights.min():.3f}; {self.weights.max():.3f}]'
                f'| {low_mx:.4f}  {y.max():.3f}'
            )
            stats += f'|'
            if self.adaptive_beta:
                # noinspection PyUnboundLocalVariable
                stats += f' {active_mass:.3f}'
            stats += f' {values.sum():.3f}  {sdr.size}'
            print(stats)

        return output_sdr

    @timed
    def _compute_batch(self, input_sdrs: SdrArray, learn: bool) -> SdrArray:
        batch_size = len(input_sdrs)
        output_sdrs = []

        xs = input_sdrs.get_batch_dense(np.arange(batch_size))
        w = self.weights

        us_raw = np.dot(w, xs.T).T

        for i in range(batch_size):
            x, u_raw = xs[i], us_raw[i]

            u = u_raw
            if self.bias_boosting:
                u = self.boost_potentials(u_raw)

            y = softmax(u, beta=self.beta)

            # Fixed threshold
            thr = self.threshold
            sdr = np.flatnonzero(y > thr)
            values = y[sdr]
            mass = values.sum()

            self.fast_output_sdr_size_trace.put(len(sdr))
            if self.filter_output == FilterOutput.HARD:
                o_sdr = sdr
            else:
                o_sdr = np.flatnonzero(y > 1e-4)

            o_values = y[o_sdr].copy()
            if self.normalize_output == NormalizeOutput.YES:
                o_values /= mass + 1e-30

            output_sdr = RateSdr(o_sdr, o_values)
            output_sdrs.append(output_sdr)
            if not learn or sdr.size == 0:
                continue

            k = self.output_sds.active_size
            top_k = None
            if self.adaptive_beta or self.negative_hebbian == NegativeHebbian.TOP_K:
                cur_k = min(k, len(sdr))
                top_k = values[np.argpartition(values, -cur_k)[-cur_k:]]

            y = y[sdr]
            if self.negative_hebbian == NegativeHebbian.RATE:
                y = y - self.output_rate[sdr]
            elif self.negative_hebbian == NegativeHebbian.TOP_K:
                y = y - np.min(top_k)
                y[y < 0] *= 0.2

            lr = self.learning_rate
            if self.adaptive_lr:
                lr = self.get_adaptive_lr(sdr)

            _u = np.expand_dims(u_raw[sdr], -1)
            _x = np.expand_dims(x, 0)
            _y = np.expand_dims(y, -1)
            lr = np.expand_dims(lr, -1)

            d_weights = _y * (_x - w[sdr] * _u)
            if self.normalize_dw:
                d_weights /= np.abs(d_weights).max() + 1e-30
            self.weights[sdr, :] += lr * d_weights

            self.radius[sdr] = self.get_radius(sdr)
            self.relative_radius[sdr] = self.get_relative_radius(sdr)
            self.slow_output_trace.put(output_sdr.values, output_sdr.sdr)

            if self.bias_boosting:
                self.biases = np.log(self.output_rate)

            if self.adaptive_beta:
                beta_lr = self.beta_lr * np.sqrt(np.mean(lr))
                active_mass = top_k.sum()

                self.fast_active_mass_trace.put(active_mass)
                avg_active_mass = self.fast_active_mass_trace.get()
                avg_active_size = self.output_active_size

                d_beta = 0.0
                if avg_active_size < k:
                    d_beta = -0.02
                elif avg_active_mass < self.min_active_mass or avg_active_mass > self.min_mass:
                    target_mass = (self.min_active_mass + self.min_mass) / 2
                    rel_mass = max(0.1, avg_active_mass / target_mass)
                    # less -> neg (neg log) -> increase beta and vice versa
                    d_beta = -np.log(rel_mass)

                if d_beta != 0.0:
                    self.beta *= np.exp(beta_lr * np.clip(d_beta, -1.0, 1.0))
                    self.beta += beta_lr * d_beta
                    self.beta = np.clip(self.beta, 1e-3, 1e+4)

                thr_lr = self.threshold_lr * np.sqrt(np.mean(lr))
                if avg_active_size < k:
                    d_thr = -1.0
                elif avg_active_size > k * (1.0 + 2 * self.output_extra):
                    d_thr = 1.0
                else:
                    d_thr = 0.1 * np.sign(avg_active_size - k * (1.0 + self.output_extra))

                self.threshold += thr_lr * d_thr
                self.threshold = max(1e-5, self.threshold)

            self.cnt += 1
            if self.print_stats_scheduler.tick():
                low_y = y[y <= thr]
                low_mx = 0. if low_y.size == 0 else low_y.max()
                stats = (
                    f'{self.avg_radius:.3f} {self.output_entropy():.3f} {self.output_active_size:.1f}'
                    f'| {self.beta:.1f} {self.threshold:.4f}'
                )
                if self.bias_boosting:
                    stats += (
                        f'| {self.biases.mean():.3f} [{self.biases.min():.3f}; {self.biases.max():.3f}]'
                    )
                stats += (
                    f'| {self.weights.mean():.3f}: [{self.weights.min():.3f}; {self.weights.max():.3f}]'
                    f'| {low_mx:.4f}  {y.max():.3f}'
                )
                stats += f'|'
                if self.adaptive_beta:
                    # noinspection PyUnboundLocalVariable
                    stats += f' {active_mass:.3f}'
                stats += f' {values.sum():.3f}  {sdr.size}'
                print(stats)

        return SdrArray(sparse=output_sdrs, sdr_size=self.output_size)

    def boost_potentials(self, u):
        b = self.biases / self.base_bias
        bb = 1.0 + np.clip(b * np.maximum(self.relative_radius, 0.) / 10, 0.0, 100.0)
        return u * bb

    def get_weight_pow_p(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.lebesgue_p - 1
        w = self.weights if ixs is None else self.weights[ixs]
        if p == 1:
            # shortcut to remove unnecessary calculations
            return w
        return np.sign(w) * (np.abs(w) ** p)

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        if ixs is None:
            w = self.weights
            return np.sqrt(np.sum(w ** 2, axis=-1))

        w = self.weights[ixs]
        return np.sqrt(np.sum(w ** 2, axis=-1))

    def get_relative_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.log2(np.maximum(r, 1e-30))

    def get_adaptive_lr(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        base_lr = self.learning_rate
        rs = self.relative_radius if ixs is None else self.relative_radius[ixs]
        return np.clip(base_lr * rs, *self.lr_range)

    @property
    def avg_radius(self):
        return self.radius.mean()

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

    def output_entropy(self):
        return entropy(self.output_rate)
