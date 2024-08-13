#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from hima.common.config.base import TConfig
from hima.common.scheduler import Scheduler
from hima.common.sdr import (
    SparseSdr, DenseSdr, RateSdr, AnySparseSdr, OutputMode, unwrap_as_rate_sdr
)
from hima.common.sdr_array import SdrArray
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.common.utils import isnone
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.pruning_controller_dense import PruningController
from hima.experiments.temporal_pooling.stp.se_utils import boosting, arg_top_k, WeightsDistribution
from hima.modules.htm.utils import abs_or_relative

if TYPE_CHECKING:
    from hima.experiments.temporal_pooling.stp.se_dense import SpatialEncoderDenseBackend


class FilterInputPolicy(Enum):
    NO = 0
    SUBTRACT_AVG = auto()
    SUBTRACT_AVG_AND_CLIP = auto()


class BoostingPolicy(Enum):
    NO = 0
    ADDITIVE = auto()
    MULTIPLICATIVE = auto()


class LearningPolicy(Enum):
    # W \cdot x
    LINEAR = 1
    # W^{p-1} \cdot x
    KROTOV = auto()


class ActivationPolicy(Enum):
    POWERLAW = 1
    EXPONENTIAL = auto()


class SpatialEncoderLayer:
    """
    A competitive network implementation from Krotov-Hopfield with several modifications.
    Source: Unsupervised learning by competing hidden units
        https://pnas.org/doi/full/10.1073/pnas.1820458116

    Modifications:
    """
    rng: Generator

    # input
    feedforward_sds: Sds
    adapt_to_ff_sparsity: bool

    # input cache
    sparse_input: SparseSdr
    dense_input: DenseSdr

    # connections
    weights_backend: SpatialEncoderDenseBackend

    pruning_controller: PruningController | None
    rf_sparsity: float

    # potentiation and learning
    learning_policy: LearningPolicy
    # [M, Q]: the number of neurons affected by hebb and anti-hebb
    learning_set: tuple[int, int]
    learning_rate: float

    # output
    output_sds: Sds
    output_mode: OutputMode
    activation_threshold: tuple[float, float, float]

    def __init__(
            self, *, seed: int, feedforward_sds: Sds, output_sds: Sds,
            adapt_to_ff_sparsity: bool,

            normalize_input_p: float = 0.0, filter_input_policy: str = 'no',
            lebesgue_p: float = 1.0, init_radius: float = 10.0,
            weights_distribution: str = 'normal',
            inhibitory_ratio: float = 0.0,

            match_p: float | None = None, boosting_policy: str = 'no',
            activation_policy: str = 'powerlaw', beta: float = 1.0, beta_lr: float = 0.01,
            # K-based extra for soft partitioning
            soft_extra: float = 1.0,
            # sum(topK) to sum(top2K) min and max ratio
            beta_active_mass: tuple[float] = (0.7, 0.9),

            learning_policy: str = 'linear', learning_rate: float = 0.01,
            adaptive_lr: bool = False, lr_range: tuple[float, float] = (0.00001, 0.1),
            # K-based learning set: K1 | (K1, K2). K1 - hebb, K2 - anti-hebb
            learning_set: int | float | tuple[int | float] = 1.0,
            anti_hebb_scale: float = 0.4,
            persistent_signs: bool = False,
            normalize_dw: bool = False,

            normalize_output: bool = False,
            # K-based extra for output
            output_extra: int | float = 0.0,

            pruning: TConfig = None,

            print_stats_schedule: int = 10_000,
            **kwargs
    ):
        print(f'kwargs: {kwargs}')
        self.rng = np.random.default_rng(seed)

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.adapt_to_ff_sparsity = adapt_to_ff_sparsity

        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode.RATE
        k = self.output_sds.active_size

        # ==> Input preprocessing
        self.input_normalization = normalize_input_p > 0.0
        self.normalize_input_p = normalize_input_p
        self.filter_input_policy = FilterInputPolicy[filter_input_policy.upper()]

        # ==> Weights backend initialization
        match_p, lebesgue_p, learning_policy = align_matching_learning_params(
            match_p, lebesgue_p, learning_policy
        )
        weights_distribution = WeightsDistribution[weights_distribution.upper()]

        # TODO: after implementation of sparse backend, add conditional initialization
        from hima.experiments.temporal_pooling.stp.se_dense import SpatialEncoderDenseBackend
        self.weights_backend = SpatialEncoderDenseBackend(
            seed=self.rng.integers(100_000_000),
            feedforward_sds=self.feedforward_sds, output_sds=self.output_sds,
            lebesgue_p=lebesgue_p, init_radius=init_radius,
            weights_distribution=weights_distribution, inhibitory_ratio=inhibitory_ratio,
            match_p=match_p,
            learning_policy=learning_policy, persistent_signs=persistent_signs,
            normalize_dw=normalize_dw, pruning=pruning
        )
        self.pos_log_radius = self.get_pos_log_radius()

        # ==> Pattern matching
        self.bias_boosting = BoostingPolicy[boosting_policy.upper()]

        # ==> K-extras
        soft_extra = abs_or_relative(soft_extra, k)
        output_extra = abs_or_relative(output_extra, k)
        k1, k2 = parse_learning_set(learning_set, k)

        # ==> Potentials soft partitioning
        # it should be high enough to cover all defined extras
        # NB: we also require at least K/2 extra items to adjust beta
        soft_extra = max(k//2, soft_extra, k2, output_extra)

        self.exact_partitioning = self.output_size <= 600
        if self.exact_partitioning:
            # simple ArgPartition is faster for small arrays
            self.soft_top_k = k + soft_extra
        else:
            # for large arrays, more advanced partitioning is faster
            #   we use sqrt partitioning of potentials to find maxes distribution
            block_size = int(np.sqrt(self.output_size))
            #   due to the approximate nature, we sub-linearize the extra to get soft-top-K value
            soft_top_k = k + round(soft_extra ** 0.7)
            #   cache full config for the approximate partitioning
            self.soft_top_k = (
                block_size, block_size * block_size,
                min(soft_top_k, block_size)
            )

        # ==> Activation [applied to soft partition]
        self.activation_policy = ActivationPolicy[activation_policy.upper()]
        # for Exp, beta is inverse temperature in the softmax
        # for Powerlaw, beta is the power in the RePU (for simplicity I use the same name)
        self.beta = beta
        self.adaptive_beta = beta_lr > 0.0
        if self.adaptive_beta:
            if self.activation_policy == ActivationPolicy.POWERLAW:
                # compared to the softmax beta, the power beta is usually smaller, hence lr is
                # scaled down to equalize settings for diff activation policies
                beta_lr /= 10.0
            self.beta_lr = beta_lr

        # sum(topK) to sum(top2K) (min, max) ratio
        self.beta_active_mass = beta_active_mass

        # ==> Hard partitioning: for learning and output selection
        self.hard_top_k = k + max(k2, output_extra)

        # ==> Learning
        self.learning_policy = learning_policy
        # global learn flag that is switched each compute, to avoid passing it through
        # the whole chain on demand. After compute it's set to False automatically
        self.learn = False
        # [:k1] - hebbian, [k:k+k2] - anti-hebbian
        self.learning_set = (k1, k2)
        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        if self.adaptive_lr:
            self.lr_range = lr_range
        self.anti_hebbian_scale = anti_hebb_scale
        self.persistent_signs = persistent_signs
        self.normalize_dw = normalize_dw

        # ==> Output
        self.normalize_output = normalize_output
        self.output_extra = output_extra

        self.cnt = 0
        self.loops = 0

        # stats collection
        slow_lr = LearningRateParam(window=40_000)
        fast_lr = LearningRateParam(window=10_000)
        self.computation_speed = MeanValue(lr=slow_lr)

        self.fast_potentials_trace = MeanValue(size=self.output_size, lr=fast_lr, initial_value=0.)
        self.fast_soft_size_trace = MeanValue(lr=fast_lr)

        self.fast_feedforward_trace = MeanValue(size=self.ff_size, lr=fast_lr, initial_value=0.)
        self.slow_feedforward_size_trace = MeanValue(lr=slow_lr)

        self.slow_output_trace = MeanValue(
            size=self.output_size, lr=slow_lr, initial_value=1.0 / self.output_size
        )
        self.fast_output_sdr_size_trace = MeanValue(lr=fast_lr, initial_value=k + self.output_extra)

        if self.adaptive_beta:
            self.fast_hard_size_trace = MeanValue(lr=fast_lr)
            self.fast_active_mass_trace = MeanValue(
                lr=fast_lr, initial_value=self.beta_active_mass[0]
            )
        self.print_stats_scheduler = Scheduler(print_stats_schedule)

    def compute_batch(self, input_sdrs: SdrArray, learn: bool = False) -> SdrArray:
        self.learn = learn
        output_sdr, run_time = self._compute_batch(input_sdrs)
        # put average time per SDR
        self.computation_speed.put(run_time / len(input_sdrs))
        self.learn = False
        return output_sdr

    def compute(self, input_sdr: AnySparseSdr, learn: bool = False) -> AnySparseSdr:
        """Compute the output SDR."""
        self.learn = learn
        output_sdr, run_time = self._compute(input_sdr)
        self.computation_speed.put(run_time)
        self.learn = False
        return output_sdr

    @timed
    def _compute(self, input_sdr: AnySparseSdr) -> AnySparseSdr:
        input_sdr = RateSdr(*unwrap_as_rate_sdr(input_sdr))

        prepr_input_sdr = self.preprocess_input(input_sdr)
        x = np.zeros(self.ff_size)
        x[prepr_input_sdr.sdr] = prepr_input_sdr.values

        # ==> Match input
        u_raw = self.match_input(x)
        u = self.apply_boosting(u_raw)

        # ==> Soft partition
        # with soft partition we take a very small subset, in [0, 10%] of the full array
        # NB: with this, we simulate taking neurons with supra-average activity
        soft_sdr = self.partition_potentials(u)
        # make a copy for better performance
        soft_u = u[soft_sdr].copy()

        # ==> Activate
        soft_y = self.activate(soft_u)

        # ==> apply hard partition (optionally) and sort activations
        # NB: both hard_sdr and y are sorted by activations in desc order
        hard_sdr, hard_y = self.partition_and_rank_activations(soft_sdr, soft_y)

        # ==> Select output
        output_sdr = self.select_output(hard_sdr, hard_y)

        if not self.learn:
            return output_sdr

        # ==> Select learning set
        sdr_learn, y_learn = self.select_learning_set(hard_sdr, hard_y)
        # ==> Learn
        self.update_dense_weights(x, sdr_learn, y_learn, u_raw)
        self.update_beta()
        self.prune_newborns()
        self.cnt += 1

        if self.print_stats_scheduler.tick():
            self.print_stats(u, output_sdr)
        # if self.cnt % 50000 == 0:
        #     self.plot_weights_distr()
        # if self.cnt % 10000 == 0:
        #     self.plot_activation_distr(sdr, u, y)

        return output_sdr

    @timed
    def _compute_batch(self, input_sdrs: SdrArray) -> SdrArray:
        batch_size = len(input_sdrs)
        output_sdrs = []

        # ==> Accept input
        prepr_input_sdrs = SdrArray(sparse=[
            self.preprocess_input(input_sdrs.sparse[i]) for i in range(batch_size)
        ], sdr_size=input_sdrs.sdr_size)

        if self.filter_input_policy == FilterInputPolicy.NO and not self.input_normalization:
            xs = input_sdrs.get_batch_dense(np.arange(batch_size))
        else:
            xs = prepr_input_sdrs.get_batch_dense(np.arange(batch_size))

        # ==> Match input
        us_raw = self.match_input(xs.T).T
        us = self.apply_boosting(us_raw)

        for i in range(batch_size):
            x = xs[i]
            u, u_raw = us[i], us_raw[i]

            # ==> Soft partition
            # with soft partition we take a very small subset, in [0, 10%] of the full array
            # NB: with this, we simulate taking neurons with supra-average activity
            soft_sdr = self.partition_potentials(u)

            # make a copy for better performance
            soft_u = u[soft_sdr].copy()

            # ==> Activate
            soft_y = self.activate(soft_u)

            # ==> apply hard partition (optionally) and sort activations
            # NB: both hard_sdr and y are sorted by activations in desc order
            hard_sdr, hard_y = self.partition_and_rank_activations(soft_sdr, soft_y)

            # ==> Select output
            output_sdr = self.select_output(hard_sdr, hard_y)
            output_sdrs.append(output_sdr)

            if not self.learn:
                continue

            # ==> Select learning set
            sdr_learn, y_learn = self.select_learning_set(hard_sdr, hard_y)
            # ==> Learn
            self.update_dense_weights(x, sdr_learn, y_learn, u_raw)
            self.cnt += 1
            # if self.cnt % 50000 == 0:
            #     self.plot_weights_distr()
            # if self.cnt % 10000 == 0:
            #     self.plot_activation_distr(sdr, u, y)

        if self.learn:
            # ==> Learn (continue)
            self.prune_newborns(batch_size)
            # single beta update with increased force (= batch size)
            self.update_beta(mu=batch_size)
            if self.print_stats_scheduler.tick(batch_size):
                self.print_stats(us[-1], output_sdrs[-1])

        return SdrArray(sparse=output_sdrs, sdr_size=self.output_size)

    def partition_potentials(self, u):
        if self.exact_partitioning:
            soft_sdr = arg_top_k(u, self.soft_top_k)
        else:
            # approximate by using max values distribution from square 2d regrouping
            b, sz, soft_k = self.soft_top_k
            partitions_maxes = u[:sz].reshape(b, b).max(axis=-1)
            # ASC order
            partitions_maxes.sort()
            # take the softK-th lowest max, where softK is specifically chosen to allow at least
            # K + ~E winners to pass through
            t = partitions_maxes[-soft_k]
            soft_sdr = np.flatnonzero(u > t)

        if self.learn:
            self.fast_soft_size_trace.put(len(soft_sdr))
        return soft_sdr

    def partition_and_rank_activations(self, soft_sdr, y):
        sz = len(soft_sdr)
        if sz > 100 and sz / self.hard_top_k > 2.0:
            # if the difference is significant, apply hard partitioning
            ixs = arg_top_k(y, self.hard_top_k)
            y = y[ixs]
            soft_sdr = soft_sdr[ixs] if soft_sdr is not None else ixs

        # rank by activations in DESC order
        ixs = np.argsort(y)[::-1]

        # apply the order
        y = y[ixs]
        sdr = soft_sdr[ixs] if soft_sdr is not None else ixs

        if self.learn and self.adaptive_beta:
            k = self.output_sds.active_size
            self.fast_hard_size_trace.put(len(sdr))
            self.fast_active_mass_trace.put(y[:k].sum())
        return sdr, y

    def activate(self, u):
        if self.activation_policy == ActivationPolicy.POWERLAW:
            y = u - u.min()
            # set small, but big enough, range, where beta is still considered 1 mainly to
            # avoid almost negligible differences at the cost of additional computations
            if not (0.9 <= self.beta <= 1.2):
                y **= self.beta
        elif self.activation_policy == ActivationPolicy.EXPONENTIAL:
            # NB: no need to subtract min before, as we have to subtract max anyway
            # for numerical stability (exp is shift-invariant)
            y = np.exp(self.beta * (u - u.max()))
        else:
            raise ValueError(f'Unsupported activation function: {self.activation_policy}')

        y = normalize(y, all_positive=True)
        return y

    def match_input(self, x):
        return self.weights_backend.match_input(x)

    def apply_boosting(self, u_raw):
        if self.bias_boosting == BoostingPolicy.NO:
            u = u_raw
        elif self.bias_boosting == BoostingPolicy.ADDITIVE:
            avg_u = self.fast_potentials_trace.get()
            u = u_raw - avg_u
            if self.learn:
                if u_raw.ndim == 2:
                    for i in range(u_raw.shape[0]):
                        self.fast_potentials_trace.put(u_raw[i])
                else:
                    self.fast_potentials_trace.put(u_raw)
        elif self.bias_boosting == BoostingPolicy.MULTIPLICATIVE:
            boosting_k = boosting(relative_rate=self.output_relative_rate, k=self.pos_log_radius)
            u = u_raw * boosting_k
        else:
            raise ValueError(f'Unsupported boosting policy: {self.bias_boosting}')

        return u

    def select_output(self, hard_sdr, hard_y):
        k, extra = self.output_sds.active_size, self.output_extra
        k_output = min(k + extra, len(hard_sdr))

        sdr = hard_sdr[:k_output].copy()
        y = hard_y[:k_output].copy()
        if self.normalize_output:
            y = normalize(y, all_positive=True)

        if k_output > 0:
            eps = 0.05 / k_output
            mask = y > eps

            sdr = sdr[mask].copy()
            y = y[mask].copy()

        output_sdr = RateSdr(sdr, y)
        if self.learn:
            self.fast_output_sdr_size_trace.put(len(output_sdr.sdr))
            self.slow_output_trace.put(output_sdr.values, output_sdr.sdr)
        return output_sdr

    def select_learning_set(self, sdr, y):
        k1, k2 = self.learning_set
        k = self.output_sds.active_size

        # select top K1 for Hebbian learning
        sdr_hebb = sdr[:k1]
        y_hebb = y[:k1]

        k2 = min(k2, len(sdr) - k)
        if k2 > 0:
            # select K2, starting from K, for Anti-Hebbian learning
            sdr_anti_hebb = sdr[k:k + k2]
            y_anti_hebb = y[k:k + k2]
        else:
            sdr_anti_hebb = np.empty(0, dtype=int)
            y_anti_hebb = np.empty(0, dtype=float)

        return (sdr_hebb, sdr_anti_hebb), (y_hebb, y_anti_hebb)

    def update_dense_weights(self, x, sdr, y, u):
        (sdr_hebb, sdr_anti_hebb), (y_hebb, y_anti_hebb) = sdr, y
        sdr = np.concatenate([sdr_hebb, sdr_anti_hebb])

        if sdr.size == 0:
            return

        y_anti_hebb = -self.anti_hebbian_scale * y_anti_hebb
        y = np.concatenate([y_hebb, y_anti_hebb])
        lr = self.get_adaptive_lr(sdr) if self.adaptive_lr else self.learning_rate
        if self.filter_input_policy == FilterInputPolicy.SUBTRACT_AVG:
            x = np.abs(x)

        self.weights_backend.update_dense_weights(x, sdr, y, u, lr=lr)
        self.pos_log_radius[sdr] = self.get_pos_log_radius(sdr)

    def preprocess_input(self, input_sdr: RateSdr):
        # ==> subtract average rates
        if self.filter_input_policy == FilterInputPolicy.NO:
            rates = input_sdr.values
        elif self.filter_input_policy == FilterInputPolicy.SUBTRACT_AVG:
            rates = input_sdr.values - self.fast_feedforward_trace.get(input_sdr.sdr)
        elif self.filter_input_policy == FilterInputPolicy.SUBTRACT_AVG_AND_CLIP:
            rates = input_sdr.values - self.fast_feedforward_trace.get(input_sdr.sdr)
            rates = np.maximum(rates, 0)
        else:
            raise ValueError(f'Unsupported filter input policy: {self.filter_input_policy}')

        # ==> normalize input
        if self.input_normalization:
            p = self.normalize_input_p
            rates = normalize(rates, p)
            raise NotImplementedError('Normalize by avg RF norm')

        if self.learn:
            self.slow_feedforward_size_trace.put(len(input_sdr.sdr))
            self.fast_feedforward_trace.put(input_sdr.values, input_sdr.sdr)

        return RateSdr(input_sdr.sdr, rates)

    def prune_newborns(self, ticks_passed: int = 1):
        self.weights_backend.prune_newborns(ticks_passed)

    def update_beta(self, mu: int = 1):
        schedule = 16
        # mu: additional learning scale (to make mu updates in one hop)
        if mu == 1:
            # sequential mode
            if self.cnt % schedule != 0:
                # beta updates throttling, update every xxx-th step
                return
            # make an update with accumulated force
            mu = schedule

        if not self.adaptive_beta:
            return

        k = self.output_sds.active_size
        k_extra = self.fast_output_sdr_size_trace.get() - k
        if k_extra < 0.4 * k:
            return

        avg_pos_log_radius = max(0.01, self.pos_log_radius.mean())
        beta_lr = mu * self.beta_lr * max(0.01, np.sqrt(avg_pos_log_radius))

        avg_active_mass = self.fast_active_mass_trace.get()
        avg_active_size = self.output_active_size

        m_low, m_high = adapt_beta_mass(k, k_extra, self.beta_active_mass)

        d_beta = 0.0
        if avg_active_size < k:
            d_beta = -0.02
        elif not (m_low <= avg_active_mass <= m_high):
            target_mass = (m_low + m_high) / 2
            rel_mass = max(0.1, avg_active_mass / target_mass)
            # less -> neg (neg log) -> increase beta and vice versa
            d_beta = -np.log(rel_mass)

        if d_beta != 0.0:
            self.beta *= np.exp(beta_lr * np.clip(d_beta, -1.0, 1.0))
            self.beta += beta_lr * d_beta
            self.beta = max(min(self.beta, 1e+5), 1e-4)

    def get_pos_log_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.log2(np.maximum(r, 1.0))

    def get_adaptive_lr(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        base_lr = self.learning_rate
        rs = self.pos_log_radius if ixs is None else self.pos_log_radius[ixs]
        return np.clip(base_lr * rs, *self.lr_range)

    def print_stats(self, u, output_sdr):
        sdr, y = unwrap_as_rate_sdr(output_sdr)
        r = self.radius.mean()
        k = self.output_sds.active_size
        active_mass = 100.0 * y[:k].sum()
        biases = self.output_rate / self.output_sds.sparsity
        w = self.weights_backend.weights
        eps = r * 0.2 / self.ff_size
        signs_w = np.sign(w)
        pos_w = 100.0 * np.count_nonzero(signs_w > eps) / w.size
        neg_w = 100.0 * np.count_nonzero(signs_w < -eps) / w.size
        zero_w = 100.0 - pos_w - neg_w
        print(
            f'R={r:.3f} H={self.output_entropy():.3f}'
            f' B={self.beta:.3f} S={self.output_active_size:.1f}'
            f' SfS={self.fast_soft_size_trace.get():.1f}'
            f'| B[{biases.min():.2f}; {biases.max():.2f}]'
            f'| U[{u.min():.3f}  {u.max():.3f}]'
            f'| W {w.mean():.3f} [{w.min():.3f}; {w.max():.3f}]'
            f' NZP[{neg_w:.0f}; {zero_w:.0f}; {pos_w:.0f}]'
            f'| Y {active_mass:.0f} {sdr.size}'
        )

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

    @property
    def lebesgue_p(self):
        return self.weights_backend.lebesgue_p

    @property
    def radius(self):
        return self.weights_backend.radius

    @property
    def ff_size(self):
        return self.feedforward_sds.size

    @property
    def ff_avg_active_size(self):
        return round(self.slow_feedforward_size_trace.get())

    @property
    def ff_avg_sparsity(self):
        return self.ff_avg_active_size / self.ff_size

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
        output_rates = self.output_rate
        return output_rates / output_rates.sum()

    def output_entropy(self):
        return entropy(self.output_rate)


def align_matching_learning_params(
        match_p: float | None, lebesgue_p: float, learning_policy: str
) -> tuple[float, float, LearningPolicy]:
    learning_policy = LearningPolicy[learning_policy.upper()]
    # check learning rule with p-norm compatibility
    assert (
            (learning_policy == LearningPolicy.LINEAR and lebesgue_p == 1.0) or
            (learning_policy == LearningPolicy.KROTOV and lebesgue_p >= 2.0)
    ), f'{learning_policy} is incompatible with p-norm {lebesgue_p}'

    # check learning rule with matching power p compatibility
    if learning_policy == LearningPolicy.LINEAR:
        # for linear learning, we allow any weights power, default is 1.0 (linear)
        match_p = isnone(match_p, 1.0)
    elif learning_policy == LearningPolicy.KROTOV:
        # for krotov learning, power is fixed to weights' norm p - 1
        induced_match_p = lebesgue_p - 1
        assert isnone(match_p, induced_match_p) == induced_match_p
        match_p = induced_match_p

    return match_p, lebesgue_p, learning_policy


def adapt_beta_mass(k, soft_extra, beta_active_mass):
    a = (soft_extra / k) ** 0.7

    def adapt_relation(x):
        nx = (1.0 - x) * a
        return x / (x + nx)

    low, high = beta_active_mass
    return adapt_relation(low), adapt_relation(high)


def normalize(x, p=1.0, all_positive=False):
    u = x if all_positive else np.abs(x)
    r = np.sum(u ** p, axis=-1) ** (1 / p) if p != 1.0 else np.sum(u, axis=-1)
    eps = 1e-30
    if x.ndim > 1:
        mask = r > eps
        return x[mask] / np.expand_dims(r[mask], -1)

    return x / r if r > eps else x


def parse_learning_set(ls, k):
    # K - is a base number of active neuron as if we used k-WTA activation/learning.
    # However, we can disentangle learning and activation, so we can have different number
    # of active and learning neurons. It is useful in some cases to propagate more than K winners
    # to the output. It can be also useful (and it does not depend on the previous case) to
    # learn additional number of near-winners with anti-Hebbian learning. Sometimes we may want
    # to propagate them to the output, sometimes not.
    # TL;DR: K is the base number, it defines a desirable number of winning active neurons. But
    # we also define:
    #  - K1 <= K: the number of neurons affected by Hebbian learning
    #  - K2: the number of neurons affected by anti-Hebbian learning, starting from K-th index

    # ls: K1 | [K1, K2]
    if not isinstance(ls, (tuple, list)):
        k1 = round(abs_or_relative(ls, k))
        k2 = 0
    elif len(ls) == 1:
        k1 = round(abs_or_relative(ls[0], k))
        k2 = 0
    elif len(ls) == 2:
        k1 = round(abs_or_relative(ls[0], k))
        k2 = round(abs_or_relative(ls[1], k))
    else:
        raise ValueError(f'Unsupported learning set: {ls}')

    k1 = min(k, k1)
    return k1, k2
