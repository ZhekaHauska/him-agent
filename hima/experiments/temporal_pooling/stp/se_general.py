#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from enum import Enum, auto

import numpy as np
import numpy.typing as npt
from numba import jit
from numpy.random import Generator

from hima.common.config.base import TConfig
from hima.common.sdr import (
    SparseSdr, DenseSdr, RateSdr, AnySparseSdr, OutputMode, unwrap_as_rate_sdr
)
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.pruning_controller_dense import PruningController
from hima.experiments.temporal_pooling.stp.sp_utils import tick, boosting
from hima.modules.htm.utils import abs_or_relative


class WeightsDistribution(Enum):
    NORMAL = 1
    UNIFORM = auto()


class FilterInputPolicy(Enum):
    NO = 0
    SUBTRACT_AVG = auto()
    SUBTRACT_AVG_AND_CLIP = auto()


class MatchPolicy(Enum):
    # W \cdot x
    LINEAR = 1
    # W^{1/2} \cdot x
    SQRT = auto()
    # W^{p-1} \cdot x
    KROTOV = auto()


class BoostingPolicy(Enum):
    NO = 0
    ADDITIVE = auto()
    MULTIPLICATIVE = auto()


class ActivationFunc(Enum):
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
    weights: npt.NDArray[float]
    lebesgue_p: float

    pruning_controller: PruningController | None
    rf_sparsity: float

    # potentiation and learning
    match_policy: MatchPolicy
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
            weights_bias: float = 0.0, weights_distribution: str = 'normal',
            inhibitory_ratio: float = 0.0, persistent_signs: bool = False,

            match_policy: str = 'linear', boosting_policy: str = 'no',
            activation_func: str = 'powerlaw', activation_policy: str = 'top_k',
            beta: float = 1.0, beta_lr: float = 0.01,
            # K-based extra for soft partitioning
            soft_extra: float = 1.0,
            # sum(topK) to sum(top2K) min and max ratio
            beta_active_mass: tuple[float] = (0.7, 0.9),

            learning_rate: float = 0.01,
            adaptive_lr: bool = False, lr_range: tuple[float, float] = (0.00001, 0.1),
            # K-based learning set: K1 | (K1, K2). K1 - hebb, K2 - anti-hebb
            learning_set: int | float | tuple[int | float] = 1.0,
            anti_hebb_scale: float = 0.4,
            normalize_dw: bool = False,

            normalize_output: bool = False,
            # K-based extra for output
            output_extra: int | float = 0.0,

            pruning: TConfig = None,
            **kwargs
    ):
        print(f'kwargs: {kwargs}')
        self.rng = np.random.default_rng(seed)

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.adapt_to_ff_sparsity = adapt_to_ff_sparsity
        self.sparse_input = np.empty(0, dtype=int)
        self.dense_input = np.zeros(self.ff_size, dtype=float)

        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode.RATE
        k = self.output_sds.active_size

        # ==> Input preprocessing
        self.input_normalization = normalize_input_p > 0.0
        self.normalize_input_p = normalize_input_p
        self.filter_input_policy = FilterInputPolicy[filter_input_policy.upper()]

        # ==> Weights initialization
        self.lebesgue_p = lebesgue_p
        self.rf_sparsity = 1.0
        self.weights_distribution = WeightsDistribution[weights_distribution.upper()]
        w_shape = (self.output_size, self.ff_size)
        if self.weights_distribution == WeightsDistribution.NORMAL:
            init_std = get_normal_std(init_radius, self.ff_size, self.lebesgue_p)
            self.weights = np.abs(self.rng.normal(loc=weights_bias, scale=init_std, size=w_shape))
        elif self.weights_distribution == WeightsDistribution.UNIFORM:
            init_std = get_uniform_std(init_radius, self.ff_size, self.lebesgue_p)
            self.weights = self.rng.uniform(weights_bias, init_std, size=w_shape)
        else:
            raise ValueError(f'Unsupported distribution: {weights_distribution}')

        self.radius = self.get_radius()
        self.pos_log_radius = self.get_log_radius()

        # make a portion of weights negative
        if inhibitory_ratio > 0.0:
            inh_mask = self.rng.binomial(1, inhibitory_ratio, size=w_shape).astype(bool)
            self.weights[inh_mask] *= -1.0

        # ==> Pattern matching
        self.match_policy = MatchPolicy[match_policy.upper()]
        if self.match_policy == MatchPolicy.SQRT:
            self.sqrt_w = self.get_weight_pow_p(p=0.5)
        elif self.match_policy == MatchPolicy.KROTOV:
            self.weights_pow_p = self.get_weight_pow_p()

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
        self.activation_func = ActivationFunc[activation_func.upper()]
        if self.activation_func in [ActivationFunc.POWERLAW, ActivationFunc.EXPONENTIAL]:
            # for Exp, beta is inverse temperature in the softmax
            # for Powerlaw, beta is the power in the RePU (for simplicity I use the same name)
            self.beta = beta
            self.adaptive_beta = beta_lr > 0.0
            if self.adaptive_beta:
                if self.activation_func == ActivationFunc.POWERLAW:
                    # compared to the softmax beta, the power beta is usually smaller, hence lr is
                    # scaled down to equalize settings for diff activation policies
                    beta_lr /= 10.0
                self.beta_lr = beta_lr

        # sum(topK) to sum(top2K) (min, max) ratio
        self.beta_active_mass = beta_active_mass

        # ==> Hard partitioning: for learning and output selection
        self.hard_top_k = k + max(k2, output_extra)

        # ==> Learning
        # [:k1] - hebbian, [k:k+k2] - anti-hebbian
        self.learning_set = (k1, k2)
        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        if self.adaptive_lr:
            self.lr_range = lr_range
        self.anti_hebbian_scale = anti_hebb_scale
        self.normalize_dw = normalize_dw
        self.persistent_signs = persistent_signs

        # ==> Output
        self.normalize_output = normalize_output
        self.output_extra = output_extra

        self.pruning_controller = None
        if pruning is not None:
            self.pruning_controller = PruningController(self, **pruning)

        self.cnt = 0
        self.loops = 0
        print(f'init_std: {init_std:.3f} | {self.avg_radius:.3f}')

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
            self.fast_active_mass_trace = MeanValue(
                lr=fast_lr, initial_value=self.beta_active_mass[0]
            )

    def compute(self, input_sdr: AnySparseSdr, learn: bool = False) -> AnySparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.computation_speed.put(run_time)
        return output_sdr

    @timed
    def _compute(self, input_sdr: AnySparseSdr, learn: bool) -> AnySparseSdr:
        self.accept_input(input_sdr, learn=learn)
        x = self.dense_input

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
        self.accept_output(output_sdr, learn=learn)

        if not learn:
            return output_sdr

        # ==> Select learning set
        sdr_learn, y_learn = self.select_learning_set(hard_sdr, hard_y)

        # ==> Learn
        self.update_dense_weights(x, sdr_learn, y_learn, u_raw)
        self.update_beta(hard_sdr, hard_y)

        self.fast_soft_size_trace.put(len(soft_sdr))
        self.fast_potentials_trace.put(u_raw)

        self.cnt += 1

        if self.pruning_controller is not None:
            self.prune_newborns()

        if self.cnt % 10000 == 0:
            self.print_stats(u, output_sdr)
        # if self.cnt % 50000 == 0:
        #     self.plot_weights_distr()
        # if self.cnt % 10000 == 0:
        #     self.plot_soft_threshold_accuracy()
        #     self.plot_activation_distr(sdr, u, y)

        return output_sdr

    def partition_potentials(self, u):
        if self.exact_partitioning:
            return arg_top_k(u, self.soft_top_k)
        else:
            # approximate by using max values distribution from square 2d regrouping
            b, sz, soft_k = self.soft_top_k
            partitions_maxes = u[:sz].reshape(b, b).max(axis=-1)
            # ASC order
            partitions_maxes.sort()
            # take the softK-th lowest max, where softK is specifically chosen to allow at least
            # K + ~E winners to pass through
            t = partitions_maxes[-soft_k]
            return np.flatnonzero(u > t)

    def partition_and_rank_activations(self, soft_sdr, y):
        sz = len(soft_sdr)
        if sz > 100 and sz / self.hard_top_k > 2.0:
            ixs = arg_top_k(y, self.hard_top_k)
            y = y[ixs].copy()

        # rank by activations in DESC order
        ixs = np.argsort(y)[::-1]

        # apply the order
        y = y[ixs]
        sdr = soft_sdr[ixs] if soft_sdr is not None else ixs
        return sdr, y

    def activate(self, u):
        if self.activation_func == ActivationFunc.POWERLAW:
            y = u - u.min()
            if abs(self.beta - 1.0) > 1e-2:
                y **= self.beta
        elif self.activation_func == ActivationFunc.EXPONENTIAL:
            # NB: no need to subtract min before, as we have to subtract max anyway
            # for numerical stability (exp is shift-invariant)
            y = np.exp(self.beta * (u - u.max()))
        else:
            raise ValueError(f'Unsupported activation function: {self.activation_func}')

        y = normalize(y, all_positive=True)

        return y

    def match_input(self, x):
        if self.match_policy == MatchPolicy.LINEAR:
            w = self.weights
        elif self.match_policy == MatchPolicy.SQRT:
            w = self.sqrt_w
        elif self.match_policy == MatchPolicy.KROTOV:
            w = self.weights_pow_p
        else:
            raise ValueError(f'Unsupported match policy: {self.match_policy}')
        return np.dot(w, x)

    def apply_boosting(self, u_raw):
        u = u_raw
        if self.bias_boosting == BoostingPolicy.ADDITIVE:
            avg_u = self.fast_potentials_trace.get()
            u = u_raw - avg_u
        elif self.bias_boosting == BoostingPolicy.MULTIPLICATIVE:
            boosting_k = boosting(relative_rate=self.output_relative_rate, k=self.pos_log_radius)
            u = u_raw * boosting_k
        return u

    def select_output(self, sdr, y):
        k, extra = self.output_sds.active_size, self.output_extra
        k_output = min(k + extra, len(sdr))

        sdr = sdr[:k_output].copy()
        y = y[:k_output].copy()

        if self.normalize_output:
            # TODO: normalize by avg norm
            y = normalize(y, all_positive=True)

        if k_output > 0:
            eps = 0.05 / k_output
            mask = y > eps

            sdr = sdr[mask]
            y = y[mask]

        return RateSdr(sdr, y)

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
        y_anti_hebb = -self.anti_hebbian_scale * y_anti_hebb
        sdr = np.concatenate([sdr_hebb, sdr_anti_hebb])
        y = np.concatenate([y_hebb, y_anti_hebb])

        if sdr.size == 0:
            return

        lr = self.get_adaptive_lr(sdr) if self.adaptive_lr else self.learning_rate
        w = self.weights[sdr]

        _x = np.expand_dims(x, 0)
        y_ = np.expand_dims(y, -1)
        lr_ = np.expand_dims(lr, -1)

        sg = np.sign(w) if self.persistent_signs else 1.0
        if self.match_policy == MatchPolicy.KROTOV:
            _u = np.expand_dims(u[sdr], -1)
            _u = np.maximum(0., _u)
            _u = _u / max(10.0, _u.max() + 1e-30)
            # Oja learning rule, Lp normalization, p >= 2
            dw = y_ * (sg * _x - w * _u)
        else:
            # Willshaw learning rule, L1 normalization
            dw = y_ * (sg * _x - w)

        if self.normalize_dw:
            dw /= np.abs(dw).max() + 1e-30

        self.weights[sdr, :] += lr_ * dw

        self.radius[sdr] = self.get_radius(sdr)
        self.pos_log_radius[sdr] = self.get_log_radius(sdr)
        self.recalculate_derivative_weights(sdr)

    def accept_input(self, sdr: AnySparseSdr, *, learn: bool):
        """Accept new input and move to the next time step"""
        sdr, rates = unwrap_as_rate_sdr(sdr)

        # forget prev SDR
        self.dense_input[self.sparse_input] = 0.

        # ==> subtract average rates
        if self.filter_input_policy == FilterInputPolicy.NO:
            x = rates
        elif self.filter_input_policy == FilterInputPolicy.SUBTRACT_AVG:
            x = rates - self.fast_feedforward_trace.get(sdr)
        elif self.filter_input_policy == FilterInputPolicy.SUBTRACT_AVG_AND_CLIP:
            x = rates - self.fast_feedforward_trace.get(sdr)
            x = np.maximum(x, 0)
        else:
            raise ValueError(f'Unsupported filter input policy: {self.filter_input_policy}')

        # ==> normalize input
        if self.input_normalization:
            p = self.normalize_input_p
            x = normalize(x, p)
            raise NotImplementedError('Normalize by avg norm')

        # set new SDR
        self.sparse_input = sdr
        self.dense_input[self.sparse_input] = x

        if not learn:
            return

        self.fast_feedforward_trace.put(rates, sdr)
        self.slow_feedforward_size_trace.put(len(sdr))

    def accept_output(self, sdr: RateSdr, *, learn: bool):
        sdr, value = sdr.sdr, sdr.values

        if not learn or sdr.shape[0] == 0:
            return

        # update winners activation stats
        self.fast_output_sdr_size_trace.put(len(sdr))
        self.slow_output_trace.put(value, sdr)

    def prune_newborns(self):
        pc = self.pruning_controller
        if not pc.is_newborn_phase:
            return
        now, pc.countdown = tick(pc.countdown)
        if not now:
            return

        sparsity, rf_size = pc.shrink_receptive_field()
        self.rf_sparsity = sparsity
        print(f'{sparsity:.4f} | {rf_size}')

        # rescale weights to keep the same norms
        rs = np.expand_dims(self.radius, -1)
        new_rs = np.expand_dims(self.get_radius(), -1)
        self.weights *= rs / new_rs

        self.recalculate_derivative_weights()

    def update_beta(self, hard_sdr, hard_y):
        if self.cnt % 10 != 0:
            # beta updates throttling, update every 10th step
            return

        k = self.output_sds.active_size
        k_extra = hard_y.size - k
        if not self.adaptive_beta or k_extra < 0.4 * k:
            return

        # count active mass
        k_active_mass = hard_y[:k].sum()
        self.fast_active_mass_trace.put(k_active_mass)

        avg_pos_log_radius = max(0.01, np.mean(self.pos_log_radius[hard_sdr]))
        beta_lr = self.beta_lr * max(0.01, np.sqrt(avg_pos_log_radius))

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

    def recalculate_derivative_weights(self, sdr=None):
        if self.match_policy == MatchPolicy.SQRT:
            w = self.sqrt_w
            new_w = self.get_weight_pow_p(sdr, p=0.5)
        elif self.match_policy == MatchPolicy.KROTOV:
            w = self.weights_pow_p
            new_w = self.get_weight_pow_p(sdr)
        else:
            return

        if sdr is None:
            w[:] = new_w
        else:
            w[sdr, :] = new_w

    def get_weight_pow_p(self, ixs: npt.NDArray[int] = None, p: float = None) -> npt.NDArray[float]:
        if p is None:
            p = self.lebesgue_p - 1
        w = self.weights if ixs is None else self.weights[ixs]
        if p == 1:
            # shortcut to remove unnecessary calculations
            return w
        return np.sign(w) * (np.abs(w) ** p)

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.lebesgue_p
        w = self.weights if ixs is None else self.weights[ixs]
        if p == 1:
            # shortcut to remove unnecessary calculations
            return np.sum(np.abs(w), axis=-1)
        return np.sum(np.abs(w) ** p, axis=-1) ** (1 / p)

    def get_log_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.log2(np.maximum(r, 1.0))

    def get_adaptive_lr(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        base_lr = self.learning_rate
        rs = self.pos_log_radius if ixs is None else self.pos_log_radius[ixs]
        return np.clip(base_lr * rs, *self.lr_range)

    def print_stats(self, u, output_sdr):
        sdr, y = unwrap_as_rate_sdr(output_sdr)
        r = self.avg_radius
        k = self.output_sds.active_size
        active_mass = 100.0 * y[:k].sum()
        biases = self.output_rate / self.output_sds.sparsity
        w = self.weights
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

    def plot_weights_distr(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        w, r = self.weights, self.radius
        w = w / np.expand_dims(r, -1)
        sns.histplot(w.flatten())
        if self.match_policy == MatchPolicy.SQRT:
            w_p = self.sqrt_w / np.expand_dims(r, -1)
            sns.histplot(w_p.flatten())
        plt.show()

    @property
    def avg_radius(self):
        return self.radius.mean()

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


@jit()
def get_active(a, size, thr):
    thr, mn_thr, mx_thr = thr
    mx_size = int(size * 4.0) + 1
    i = 0
    k = 2.0
    sdr = sdr_mx = np.empty(0, np.integer)

    while i <= 3:
        i += 1
        if sdr_mx.size == 0:
            sdr = np.flatnonzero(a >= thr)
        else:
            sdr = sdr_mx[np.flatnonzero(a[sdr_mx] >= thr)]

        if sdr.size < size:
            mx_thr = thr
            thr = (thr + mn_thr) / k
        elif sdr.size > mx_size:
            mn_thr = thr
            thr = (thr + mx_thr) / k
            sdr_mx = sdr
        else:
            return sdr, thr, (i, False, mn_thr, mx_thr)

    if sdr.size < size:
        if sdr_mx.size > size:
            return sdr_mx, mn_thr, (i, False, mn_thr, mx_thr)
        thr = 0.0
        return np.flatnonzero(a > thr), thr, (i, True, thr, mx_thr)

    # sdr.size > mx_size
    return sdr, mn_thr, (i, True, thr, mx_thr * 2)


@jit()
def get_active2(a, size, thr):
    thr, mn_thr, mx_thr = thr
    mx_size = int(size * 2.5) + 1
    i = 0
    k = 2.0
    sdr = sdr_mx = np.empty(0, np.integer)

    while i <= 5:
        i += 1
        sdr = np.flatnonzero(a >= thr)

        if sdr.size < size:
            mx_thr = thr
            thr = (thr + mn_thr) / k
            continue

        sdr_mx = sdr_mx[sdr] if sdr_mx.size != 0 else sdr

        if sdr.size > mx_size:
            mn_thr = thr
            thr = (thr + mx_thr) / k
            a = a[sdr].copy()
        else:
            return sdr_mx, thr, (i, False, mn_thr, mx_thr)

    if sdr.size < size:
        if sdr_mx.size > 0:
            return sdr_mx, mn_thr, (i, False, mn_thr, mx_thr)
        thr = mn_thr
        return np.flatnonzero(a >= thr), thr, (i, True, thr, mx_thr)

    # sdr.size > mx_size
    return sdr, mn_thr, (i, True, thr, mx_thr * 2)


def get_distr_std(distr: WeightsDistribution, required_r, n_samples: int, p: float) -> float:
    if distr == WeightsDistribution.NORMAL:
        return get_normal_std(required_r, n_samples, p)
    elif distr == WeightsDistribution.UNIFORM:
        return get_uniform_std(required_r, n_samples, p)
    else:
        raise ValueError(f'Unsupported distribution: {distr}')


def get_uniform_std(required_r, n_samples, p):
    alpha = 2 / n_samples
    alpha = alpha ** (1 / p)
    return required_r * alpha


def get_normal_std(required_r, n_samples: int, p: float) -> float:
    alpha = np.pi / (2 * n_samples)
    alpha = alpha ** (1 / p)
    return required_r * alpha


def adapt_beta_mass(k, soft_extra, beta_active_mass):
    a = (soft_extra / k) ** 0.7

    def adapt_relation(x):
        nx = (1.0 - x) * a
        return x / (x + nx)

    low, high = beta_active_mass
    return adapt_relation(low), adapt_relation(high)

def exp_x(
        x: npt.NDArray[float], *, beta: float, axis: int = -1
) -> npt.NDArray[float]:
    """
    Compute softmax values for a vector `x` with a given temperature or inverse temperature.
    The softmax operation is applied over the last axis by default, or over the specified axis.
    """
    beta = max(min(beta, 1e+5), 1e-4)
    # TODO: check if subtracting max val is unnecessary
    return np.exp((x - np.max(x, axis=axis, keepdims=True)) * beta)


def normalize(x, p=1.0, all_positive=False):
    u = x if all_positive else np.abs(x)
    r = np.sum(u ** p, axis=-1) if p != 1.0 else np.sum(u, axis=-1)
    eps = 1e-30
    if x.ndim > 1:
        mask = r > eps
        return x[mask] / np.expand_dims(r[mask], -1)

    return x / r if r > eps else x


def arg_top_k(x, k):
    k = min(k, x.size)
    return np.argpartition(x, -k)[-k:]


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


@jit
def nb_choice(rng, p):
    """
    Choose a sample from N values with weights p (they could be non-normalized).
    """
    # Get cumulative weights
    acc_w = np.cumsum(p)
    # Total of weights
    mx_w = acc_w[-1]
    r = mx_w * rng.random()
    # Get corresponding index
    ind = np.searchsorted(acc_w, r, side='right')
    return ind


@jit()
def get_important_simple(a, thr, size):
    i = 1
    k = 4.0
    while i <= 3:
        sdr = np.flatnonzero(a > thr)
        if sdr.size < size:
            thr /= k
        else:
            return i, thr, sdr
        i += 1
    thr = 0.0
    return i, thr, np.flatnonzero(a > thr)
