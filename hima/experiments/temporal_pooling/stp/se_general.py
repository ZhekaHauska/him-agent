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
from hima.experiments.temporal_pooling.stp.sp_utils import tick


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


class ActivationPolicy(Enum):
    LINEAR = 1
    POWERLAW = auto()
    EXPONENTIAL = auto()


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
    learning_set: LearningSet
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
            inhibitory_ratio: float = 0.0,

            match_policy: str = 'linear', bias_boosting: bool = False,
            activation_policy: str = 'linear',
            threshold: float = 0.0, threshold_lr: float = 0.0001,
            min_active_mass: float = None, min_mass: float = None,
            beta: float = 1.0, beta_lr: float = 0.01,

            learning_rate: float = 0.01, learning_set: str = 'all',
            adaptive_lr: bool = False, lr_range: tuple[float, float] = (0.00001, 0.1),
            normalize_dw: bool = False,
            negative_hebbian: str = 'no', neg_hebb_delta: float = 0.4,

            filter_output: str = 'soft', output_extra: float = 0.5, normalize_output: str = 'no',

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

        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        if self.adaptive_lr:
            self.lr_range = lr_range

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
        self.log_radius = self.get_log_radius()

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

        self.bias_boosting = bias_boosting

        # ==> Activation
        self.activation_policy = ActivationPolicy[activation_policy.upper()]
        self.soft_threshold = threshold
        self.threshold = threshold
        self.adaptive_threshold = threshold_lr > 0.0
        if self.adaptive_threshold:
            self.threshold_lr = threshold_lr
            self.threshold = min(1 / self.output_sds.size, self.output_sds.active_size ** (-2))

        self.min_active_mass = min_active_mass
        self.min_mass = min_mass

        if self.activation_policy in [ActivationPolicy.POWERLAW, ActivationPolicy.EXPONENTIAL]:
            # for Exp, beta is inverse temperature in the softmax
            # for Powerlaw, beta is the power in the RePU (for simplicity I use the same name)
            self.beta = beta
            self.adaptive_beta = beta_lr > 0.0
            if self.adaptive_beta:
                self.beta_lr = beta_lr
            if self.activation_policy == ActivationPolicy.POWERLAW:
                # compared to the softmax beta, the power beta is usually smaller, hence lr is
                # scaled down to equalize settings for diff activation policies
                self.beta_lr /= 10.0

        # ==> Learning
        self.learning_set = LearningSet[learning_set.upper()]
        self.normalize_dw = normalize_dw
        self.negative_hebbian = NegativeHebbian[negative_hebbian.upper()]
        if learning_set == LearningSet.PAIR:
            # NB: otherwise, learning diverges — weights grow to infinity
            neg_hebb_delta = neg_hebb_delta ** 2
        self.d_hebb = np.array([1.0, -neg_hebb_delta])

        # ==> Output
        self.filter_output = FilterOutput[filter_output.upper()]
        self.normalize_output = NormalizeOutput[normalize_output.upper()]
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

        self.fast_feedforward_trace = MeanValue(size=self.ff_size, lr=fast_lr, initial_value=0.)
        self.slow_feedforward_size_trace = MeanValue(lr=slow_lr)
        self.fast_hard_size_trace = MeanValue(lr=fast_lr)

        self.fast_potentials_trace = MeanValue(size=self.output_size, lr=fast_lr, initial_value=0.)

        self.slow_output_trace = MeanValue(
            size=self.output_size, lr=slow_lr, initial_value=self.output_sds.sparsity
        )
        self.fast_output_sdr_size_trace = MeanValue(
            lr=fast_lr, initial_value=self.output_sds.active_size
        )

        if self.activation_policy == ActivationPolicy.EXPONENTIAL and self.adaptive_beta:
            self.fast_mass_trace = MeanValue(lr=fast_lr, initial_value=self.min_mass)
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
        if learn:
            self.fast_potentials_trace.put(u_raw)

        # ==> Activate
        k = self.output_sds.active_size

        u = u_raw
        if self.bias_boosting:
            avg_u = self.fast_potentials_trace.get()
            u = u_raw - avg_u

        if self.activation_policy == ActivationPolicy.LINEAR:
            sdr = np.flatnonzero(u >= self.soft_threshold)
            y = normalize(u[sdr] - self.soft_threshold)
            if learn:
                self.update_soft_threshold(sdr)
            top_2k = min(2 * k, sdr.size)
            top_2k_ix = np.argpartition(y, -top_2k)[-top_2k:]
            sdr = sdr[top_2k_ix].copy()
            y = y[top_2k_ix].copy()
        elif self.activation_policy == ActivationPolicy.POWERLAW:
            sdr = np.flatnonzero(u > 0)
            # sign * abs
            y = u[sdr] ** self.beta
        elif self.activation_policy == ActivationPolicy.EXPONENTIAL:
            y = softmax(u, beta=self.beta)
            thr = self.threshold
            sdr = np.flatnonzero(y > thr)
            y = y[sdr]
        else:
            raise ValueError(f'Unsupported activation policy: {self.activation_policy}')

        # ==> Select output
        # if learn and self.cnt % 500 == 0:
        #     print(self.threshold)
        #     import matplotlib.pyplot as plt
        #     # plt.hist(u, bins=50)
        #     plt.hist(y, bins=50)
        #     plt.show()

        if self.filter_output == FilterOutput.HARD:
            top_k = min(k, sdr.size)
            top_k_ix = np.argpartition(y, -top_k)[-top_k:]
            o_sdr = sdr[top_k_ix].copy()
            o_y = y[top_k_ix].copy()
        else:
            o_sdr = sdr.copy()
            o_y = y.copy()

        if self.normalize_output == NormalizeOutput.YES:
            o_y = normalize(o_y)

        output_sdr = RateSdr(o_sdr, o_y)
        self.accept_output(output_sdr, learn=learn)

        if not learn:
            return output_sdr

        # ==> Select learning set
        n_winners = sdr.size
        if n_winners == 0:
            ixs, y_h = self.get_best_from_inactive(u)
        elif self.learning_set == LearningSet.PAIR:
            ixs, y_h = self.sample_learning_pair(sdr, y)
        elif self.learning_set == LearningSet.ALL:
            ixs, y_h = self.get_learning_set(sdr, y)
        else:
            raise ValueError(f'Unsupported learning set: {self.learning_set}')

        # ==> Learn
        self.update_dense_weights(ixs, x, y_h)
        self.update_activation_threshold()

        self.cnt += 1

        if self.pruning_controller is not None:
            self.prune_newborns()

        if self.cnt % 10000 == 0:
            self.print_stats(u, sdr, y)
        # if self.cnt % 50000 == 0:
        #     self.plot_weights_distr()

        return output_sdr

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

    def update_dense_weights(self, sdr, x, y):
        lr = self.learning_rate
        if self.adaptive_lr:
            lr = lr * self.get_adaptive_lr(sdr)
        w = self.weights[sdr]

        _x = np.expand_dims(x, 0)
        y_ = np.expand_dims(y, -1)
        lr_ = np.expand_dims(lr, -1)

        # Oja's Hebbian learning rule for L1
        # NB: brackets set order of operations to increase performance
        self.weights[sdr, :] += (lr_ * y_) * (_x - np.abs(w))

        self.radius[sdr] = self.get_radius(sdr)
        self.log_radius[sdr] = self.get_log_radius(sdr)
        if self.match_policy == MatchPolicy.SQRT:
            self.sqrt_w[sdr, :] = self.get_weight_pow_p(sdr, p=0.5)

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

    def get_learning_set(self, sdr, y):
        n_winners = sdr.size
        ac_size = self.output_sds.active_size
        assert n_winners > 0

        top_2k = min(2 * ac_size, n_winners)
        top_2k_ix = np.argpartition(y, -top_2k)[-top_2k:]
        top_2k_ix = top_2k_ix[np.argsort(y[top_2k_ix])]

        # select top K for excitation
        best_k = min(ac_size, n_winners)
        best_k_ix = top_2k_ix[-best_k:]
        ixs = best_k_ix
        y_h = self.d_hebb[0] * y[best_k_ix]

        if top_2k > ac_size:
            # select second top K for inhibition
            worst_k = top_2k - ac_size
            worst_k_ix = top_2k_ix[:worst_k]
            ixs = np.concatenate([ixs, worst_k_ix])
            y_h = np.concatenate([y_h, self.d_hebb[1] * y[worst_k_ix]])

        ixs = sdr[ixs]
        return ixs, y_h

    def sample_learning_pair(self, sdr, y):
        n_winners = sdr.size
        ac_size = self.output_sds.active_size
        assert n_winners > 0

        top_2k = min(2 * ac_size, n_winners)
        top_2k_ix = np.argpartition(y, -top_2k)[-top_2k:]
        top_2k_ix = top_2k_ix[np.argsort(y[top_2k_ix])]

        # sample single from top K [weighted by their activation] for excitation
        best_k = min(ac_size, n_winners)
        best_k_ix = top_2k_ix[-best_k:]
        excitation_ix = nb_choice(self.rng, y[best_k_ix])
        ixs = [excitation_ix]

        if top_2k > ac_size:
            worst_k = top_2k - ac_size
            # sample single from second top K [weighted by their activation] for inhibition
            worst_k_ix = top_2k_ix[:worst_k]
            inhibition_ix = nb_choice(self.rng, y[worst_k_ix])
            ixs.append(inhibition_ix)

        ixs = np.array(ixs, dtype=int)
        y_hebb = self.d_hebb[:ixs.size] * y[ixs]
        ixs = sdr[ixs]
        return ixs, y_hebb

    def get_best_from_inactive(self, u):
        ixs = np.array([np.argmax(u)], dtype=int)
        y_h = self.d_hebb[:1]
        return ixs, y_h

    def update_soft_threshold(self, sdr):
        k = self.output_sds.active_size
        lr = self.threshold_lr
        size = len(sdr)

        d_thr = 0.01 if size > k else size - k
        self.soft_threshold += lr * d_thr
        # self.soft_threshold = max(1e-10, self.soft_threshold)

    def update_activation_threshold(self):
        if self.activation_policy == ActivationPolicy.LINEAR:
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

    def boost_potentials(self, u):
        b = self.biases / self.base_bias
        bb = 1.0 + np.clip(b * np.maximum(self.log_radius, 0.) / 10, 0.0, 100.0)
        return u * bb

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
        return np.log2(np.maximum(r, 1e-30))

    def get_adaptive_lr(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        base_lr = self.learning_rate
        rs = self.log_radius if ixs is None else self.log_radius[ixs]
        return np.clip(base_lr * rs, *self.lr_range)

    def print_stats(self, u, sdr, y):
        sorted_values = np.sort(y)
        ac_size = self.output_sds.active_size
        active_mass = sorted_values[-ac_size:].sum()
        biases = self.output_rate / self.output_sds.sparsity
        w = self.weights
        non_zero_mass = np.count_nonzero(np.abs(w) > 0.2 / self.ff_size) / w.size
        sft_thr, thr = self.soft_threshold, self.threshold
        print(
            f'R={self.avg_radius:.3f} H={self.output_entropy():.3f} S={self.output_active_size:.1f}'
            f'| T [{sft_thr*100:.3f}; {thr*100:.2f}]'
            f'| B [{biases.min():.2f}; {biases.max():.2f}]'
            f'| U {u.min():.3f}  {u.max():.3f}'
            f'| W {w.mean():.3f} [{w.min():.3f}; {w.max():.3f}]  NZ={non_zero_mass:.2f}'
            f'| Y {active_mass:.3f} {sdr.size}'
        )

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
        target_rate = self.output_sds.sparsity
        return self.output_rate / target_rate

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


def normalize(x, p=1.0):
    r = np.sum(np.abs(x) ** p, axis=-1)
    eps = 1e-5
    if x.ndim > 1:
        mask = r > eps
        return x[mask] / np.expand_dims(r[mask], -1)

    return x / r if r > eps else x


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
