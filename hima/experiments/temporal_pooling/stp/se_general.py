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
from hima.common.utils import softmax
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
    LINEAR = 1
    # W^{1/2}
    SQRT = auto()
    # W^{p-1}
    KROTOV = auto()


class BoostingPolicy(Enum):
    NO = 0
    ADDITIVE = auto()
    MULTIPLICATIVE = auto()


class ActivationFunc(Enum):
    POWERLAW = 1
    EXPONENTIAL = auto()


class ActivationPolicy(Enum):
    TOP_K = 1
    THRESHOLD_F = auto()
    THRESHOLD_LINEAR = auto()


class LearningSet(Enum):
    ALL = 1
    PAIR = auto()


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


# TODO:
#   1) Consider replacing softmax before thresholding with a thresholding then softmax,
#       but it will require input normalization
#   2) Make a version for Krotov-SoftHebb learning (top 1)


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
            threshold: float = 0.0, threshold_lr: float = 0.0001,
            min_active_mass: float = None, min_mass: float = None,

            learning_rate: float = 0.01,
            adaptive_lr: bool = False, lr_range: tuple[float, float] = (0.00001, 0.1),
            # M | [M, Q] | [M, P, Q]: M - hebb, Q - anti-hebb, P - starting index for anti-hebb
            # ints or floats — absolute or relative
            learning_set: int | float | tuple[int | float] = 1.0,
            anti_hebb_scale: float = 0.4,
            normalize_dw: bool = False,

            normalize_output: str = 'no', output_extra: int | float = 0.0,

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

        # ==> Activation
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

        self.activation_policy = ActivationPolicy[activation_policy.upper()]
        self.soft_threshold = threshold
        self.threshold = threshold
        self.adaptive_threshold = threshold_lr > 0.0
        if self.adaptive_threshold:
            self.threshold_lr = threshold_lr
            self.threshold = min(1 / self.output_sds.size, self.output_sds.active_size ** (-2))

        self.min_active_mass = min_active_mass
        self.min_mass = min_mass

        # ==> Learning
        self.learning_set = parse_learning_set(learning_set, self.output_sds.active_size)
        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        if self.adaptive_lr:
            self.lr_range = lr_range
        self.anti_hebbian_scale = anti_hebb_scale
        self.normalize_dw = normalize_dw
        self.persistent_signs = persistent_signs

        # ==> Output
        self.normalize_output = NormalizeOutput[normalize_output.upper()]
        self.output_extra = abs_or_relative(output_extra, self.output_sds.active_size)

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

        self.fast_feedforward_trace = MeanValue(size=self.ff_size, lr=fast_lr, initial_value=0.)
        self.slow_feedforward_size_trace = MeanValue(lr=slow_lr)
        self.fast_hard_size_trace = MeanValue(lr=fast_lr)

        self.fast_potentials_trace = MeanValue(size=self.output_size, lr=fast_lr, initial_value=0.)

        self.slow_output_trace = MeanValue(
            size=self.output_size, lr=slow_lr, initial_value=1.0 / self.output_size
        )
        self.fast_output_sdr_size_trace = MeanValue(
            lr=fast_lr, initial_value=self.output_sds.active_size
        )

        if self.adaptive_beta:
            self.fast_active_mass_trace = MeanValue(lr=fast_lr, initial_value=self.min_active_mass)

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

        # ==> Activate
        sdr, y = self.activate(u)

        # ==> Select output
        output_sdr = RateSdr(sdr, y)
        self.accept_output(output_sdr, learn=learn)

        if not learn:
            return output_sdr

        # ==> Select learning set
        if sdr.size == 0:
            sdr_hebb, y_hebb = self.get_best_from_inactive(u)
        else:
            sdr_hebb, y_hebb = self.get_learning_set(sdr, y)

        # ==> Learn
        self.update_dense_weights(x, sdr_hebb, y_hebb, u_raw)
        self.update_beta(sdr, y)
        self.update_activation_threshold()

        self.fast_potentials_trace.put(u_raw)

        self.cnt += 1

        if self.pruning_controller is not None:
            self.prune_newborns()

        if self.cnt % 10000 == 0:
            self.print_stats(u, sdr, y)
        # if self.cnt % 50000 == 0:
        #     self.plot_weights_distr()
        # if self.cnt % 10000 == 0:
        #     self.plot_activation_distr(sdr, u, y)

        return output_sdr

    def activate(self, u):
        if self.activation_policy == ActivationPolicy.TOP_K:
            sdr, y = self.activate_top_k(u)
        else:
            sdr, y = self.activate_by_threshold(u)

        if self.normalize_output == NormalizeOutput.YES:
            # TODO: normalize by avg
            y = normalize(y, all_positive=True)

        return sdr, y

    def activate_top_k(self, u):
        # 1) F(u) -> Top K -> -u[k] -> normalize (avg) -> Output
        #   | learn top M/M+Q, where M,Q <= K/2
        k = self.output_sds.active_size
        sdr = np.flatnonzero(u > 0)
        y = u[sdr]
        e = self.output_extra
        ixs = arg_top_k(y, k + e + 1)
        sdr = sdr[ixs]
        y = y[ixs]
        if sdr.size > 1:
            y = y - y.min()
            mask = y > 0
            sdr = sdr[mask]
            y = y[mask]
        y = self.activate_f(y)
        return sdr, y

    def activate_by_threshold(self, u):
        # 2) Active SDR: F(clip[u - Soft]) & Normalize -> Slice by Hard
        #   | Output Rates: a) origin Activation -> normalize (avg);
        #       b) Linear: clip[u-Soft] & normalize (avg)
        #   | Learning: use output rates, top M/M+Q, where M,Q <= K/2
        u = u - self.soft_threshold
        sdr = np.flatnonzero(u > 0)
        v = self.activate_f(u[sdr])
        v = normalize(v, all_positive=True)
        mask = v > self.threshold
        sdr = sdr[mask]
        nonlinear_output = self.activation_policy == ActivationPolicy.THRESHOLD_F
        y = v[mask] if nonlinear_output else u[sdr]
        return sdr, y

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

    def activate_f(self, u):
        if self.activation_func == ActivationFunc.POWERLAW:
            if self.beta == 1.0:
                return u
            return u ** self.beta
        elif self.activation_func == ActivationFunc.EXPONENTIAL:
            return exp_x(u, beta=self.beta)
        else:
            raise ValueError(f'Unsupported activation function: {self.activation_func}')

    def update_dense_weights(self, x, sdr, y, u):
        lr = self.get_adaptive_lr(sdr) if self.adaptive_lr else self.learning_rate
        w = self.weights[sdr]

        _x = np.expand_dims(x, 0)
        y_ = np.expand_dims(y, -1)
        lr_ = np.expand_dims(lr, -1)

        sg = np.sign(w) if self.persistent_signs else 1.0
        if self.match_policy == MatchPolicy.KROTOV:
            _u = np.expand_dims(u[sdr], -1)
            dw = y_ * (sg * _x - w * _u)
        else:
            dw = y_ * (sg * _x - w)

        if self.normalize_dw:
            dw /= np.abs(dw).max() + 1e-30

        # Oja's Hebbian learning rule for L1
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

    def accept_output(self, sdr: AnySparseSdr, *, learn: bool):
        sdr, value = unwrap_as_rate_sdr(sdr)

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

    def get_learning_set(self, sdr, y):
        n_winners = sdr.size
        k = self.output_sds.active_size
        m, q = self.learning_set
        assert n_winners > 0

        sorted_ixs = np.argsort(y)[::-1]

        # selecting top M for hebbian reinforcement
        top_m = min(m, n_winners)
        top_m_ix = sorted_ixs[:top_m]

        ixs = top_m_ix
        y_hebb = y[top_m_ix]

        q = min(q, n_winners - k)
        if q > 0:
            # selecting top Q starting from P-th index for anti-hebbian inhibition
            top_q_anti_ix = sorted_ixs[k:k+q]
            alpha = -self.anti_hebbian_scale

            ixs = np.concatenate([ixs, top_q_anti_ix])
            y_hebb = np.concatenate([y_hebb, alpha * y[top_q_anti_ix]])

        ixs = sdr[ixs]
        return ixs, y_hebb

    @staticmethod
    def get_best_from_inactive(u):
        ixs = np.array([np.argmax(u)], dtype=int)
        y_h = np.array([1.0])
        return ixs, y_h

    def update_beta(self, sdr, y):
        if not self.adaptive_beta:
            return

        k = self.output_sds.active_size
        if self.activation_policy == ActivationPolicy.TOP_K and self.output_extra < k:
            return

        # count active mass
        active_mass = y[arg_top_k(y, k)].sum()
        self.fast_active_mass_trace.put(active_mass)

        avg_pos_log_radius = max(0.01, np.mean(self.pos_log_radius[sdr]))
        beta_lr = self.beta_lr * max(0.01, np.sqrt(avg_pos_log_radius))
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
            self.beta = max(min(self.beta, 1e+5), 1e-4)

    def update_soft_threshold(self, sdr):
        if self.activation_policy == ActivationPolicy.TOP_K:
            return

        k = self.output_sds.active_size
        lr = self.threshold_lr
        size = len(sdr)

        d_thr = 0.01 if size > k else size - k
        self.soft_threshold += lr * d_thr

    def update_activation_threshold(self):
        if self.activation_policy == ActivationPolicy.TOP_K:
            return

        k = self.output_sds.active_size
        lr = self.threshold_lr
        avg_active_size = self.fast_hard_size_trace.get()
        if avg_active_size < k:
            d_thr = avg_active_size - k
        elif avg_active_size > k * (1.0 + 2 * self.output_extra):
            d_thr = 1.0
        else:
            d_thr = 0.1 * np.sign(avg_active_size - k * (1.0 + self.output_extra))

        self.threshold += lr * d_thr
        self.threshold = max(1e-6, self.threshold)

    def recalculate_derivative_weights(self, sdr=None):
        if self.match_policy == MatchPolicy.SQRT:
            self.sqrt_w[sdr, :] = self.get_weight_pow_p(sdr, p=0.5)
        elif self.match_policy == MatchPolicy.KROTOV:
            self.weights_pow_p[sdr, :] = self.get_weight_pow_p(sdr)

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

    def print_stats(self, u, sdr, y):
        sorted_values = np.sort(y)
        ac_size = self.output_sds.active_size
        active_mass = 100.0 * sorted_values[-ac_size:].sum()
        biases = self.output_rate / self.output_sds.sparsity
        w = self.weights
        eps = 0.2 / self.ff_size
        signs_w = np.sign(w)
        pos_w = 100.0 * np.count_nonzero(signs_w > eps) / w.size
        neg_w = 100.0 * np.count_nonzero(signs_w < -eps) / w.size
        zero_w = 100.0 - pos_w - neg_w
        sft_thr, thr = self.soft_threshold, self.threshold
        print(
            f'R={self.avg_radius:.3f} H={self.output_entropy():.3f}'
            f' B={self.beta:.3f} S={self.output_active_size:.1f}'
            f'| T[{sft_thr*100:.3f}; {thr*100:.2f}]'
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
        # _u = softmax(u, beta=10.0)
        # plt.hist(_u, bins=50)
        # plt.vlines([_u[kth], _u[eth]], 0, 20, colors=['r', 'y'])
        # plt.show()
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
    mx_size = int(size * 2.5) + 1
    i = 0
    k = 2.0
    sdr = sdr_mx = np.empty(0, np.integer)

    while i <= 8:
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
    #  - M <= K: the number of neurons affected by Hebbian learning
    #  - Q: the number of neurons affected by anti-Hebbian learning, starting from K-th index

    # ls: M | [M, Q]
    if not isinstance(ls, (tuple, list)):
        m = round(abs_or_relative(ls, k))
        q = 0
    elif len(ls) == 1:
        m = round(abs_or_relative(ls[0], k))
        q = 0
    elif len(ls) == 2:
        m = round(abs_or_relative(ls[0], k))
        q = round(abs_or_relative(ls[1], k))
    else:
        raise ValueError(f'Unsupported learning set: {ls}')

    m = min(k, m)
    return m, q


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
