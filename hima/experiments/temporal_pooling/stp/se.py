#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from enum import Enum, auto

import numba
import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from hima.common.sdr import SparseSdr, DenseSdr
from hima.common.sdrr import RateSdr, AnySparseSdr, OutputMode, split_sdr_values
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import entropy


class MatchPolicy(Enum):
    LINEAR = 1
    SQRT = auto()


class LearningSet(Enum):
    ALL = 1
    PAIR = auto()


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
    # input cache
    sparse_input: SparseSdr
    dense_input: DenseSdr

    # connections
    weights: npt.NDArray[float]
    lebesgue_p: float

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
            match_p: str,
            learning_rate: float, learning_set: str,
            init_radius: float, neg_hebb_delta: float,
            **kwargs
    ):
        print(f'kwargs: {kwargs}')

        self.rng = np.random.default_rng(seed)
        self.comp_name = None

        self.feedforward_sds = Sds.make(feedforward_sds)

        self.sparse_input = np.empty(0, dtype=int)
        self.dense_input = np.zeros(self.ff_size, dtype=float)
        self.is_empty_input = True
        self.match_policy = MatchPolicy[match_p.upper()]

        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode.RATE

        self.learning_rate = learning_rate
        self.learning_set = LearningSet[learning_set.upper()]

        self.lebesgue_p = 1.0
        if learning_set == LearningSet.PAIR:
            # NB: otherwise, learning diverges — weights grow to infinity
            neg_hebb_delta /= 4
        self.d_hebb = np.array([1.0, -neg_hebb_delta])

        shape = (self.output_size, self.ff_size)
        init_std = get_normal_std(init_radius, self.ff_size, self.lebesgue_p)
        self.weights = np.abs(self.rng.normal(loc=0.0, scale=init_std, size=shape))
        # make a portion of weights negative
        self.weights[self.rng.integers(0, shape[0], size=shape[0]//5)] *= -1
        if self.match_policy == MatchPolicy.SQRT:
            self.sqrt_w = self.get_square_root_weights()

        self.radius = self.get_radius()
        self.relative_radius = self.get_relative_radius()

        self.activation_threshold = (0.0, 0.0, init_std)
        # LR anneal to 0.0001
        self.lr_activation_threshold = 0.01

        self.cnt = 0
        self.loops = 0
        print(f'init_std: {init_std:.3f} | {self.avg_radius:.3f}')

        # stats collection
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

        x, w = self.dense_input, self.weights
        if self.match_policy == MatchPolicy.LINEAR:
            u = np.dot(w, x)
        elif self.match_policy == MatchPolicy.SQRT:
            u = np.dot(self.sqrt_w, x)
        else:
            raise ValueError(f'Unsupported match policy: {self.match_policy}')

        ac_size = self.output_sds.active_size
        sdr, thr, activation_info = get_active(u, ac_size, self.activation_threshold)
        n_winners = sdr.size
        y = u[sdr] - thr
        y_r = y.sum()
        if n_winners > 0 and y_r > 0:
            y = y / y_r

        output_sdr = RateSdr(sdr, y)
        self.accept_output(output_sdr, learn=learn)

        if not learn:
            return output_sdr

        if n_winners == 0:
            ixs, y_h = self.get_best_from_inactive(u)
        elif self.learning_set == LearningSet.PAIR:
            ixs, y_h = self.sample_learning_pair(sdr, y)
        elif self.learning_set == LearningSet.ALL:
            ixs, y_h = self.get_learning_set(sdr, y)
        else:
            raise ValueError(f'Unsupported learning set: {self.learning_set}')

        self.update_dense_weights(ixs, x, y_h)

        self.update_activation_threshold(activation_info, thr)
        self.cnt += 1

        if self.cnt % 10000 == 0:
            # zero out the least important weights
            w = self.weights
            w[np.abs(w) < 0.2 * self.avg_radius/self.ff_size] = 0.0

        if self.cnt % 5000 == 0:
            self.print_stats(u, sdr, y)
        if self.cnt % 50000 == 0:
            self.plot_weights_distr()

        return output_sdr

    def update_dense_weights(self, ixs, x, y_hebb):
        lr = self.learning_rate
        lr = lr * self.relative_radius[ixs] + 0.0001
        w = self.weights[ixs]

        _x = np.expand_dims(x, 0)
        y_h_ = np.expand_dims(y_hebb, -1)
        lr_ = np.expand_dims(lr, -1)

        # Oja's Hebbian learning rule for L1
        # NB: brackets set order of operations to increase performance
        self.weights[ixs, :] += (lr_ * y_h_) * (_x * np.sign(w) - w)

        self.radius[ixs] = self.get_radius(ixs)
        self.relative_radius[ixs] = self.get_relative_radius(ixs)
        if self.match_policy == MatchPolicy.SQRT:
            self.sqrt_w[ixs, :] = self.get_square_root_weights(ixs)

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
        self.slow_output_sdr_size_trace.put(len(sdr))
        self.slow_output_trace.put(value, sdr)

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

    def update_activation_threshold(self, activation_info, new_thr):
        loops, miss, new_min_thr, new_max_thr = activation_info
        thr, min_thr, max_thr = self.activation_threshold

        miss_k = 10.0 if miss else 1.0
        lr = self.lr_activation_threshold * (loops ** 0.5)
        d_thr = new_thr - thr
        d_mn_thr = (miss_k ** 1.3) * (new_min_thr - min_thr)
        d_mx_thr = miss_k * (new_max_thr - max_thr)

        thr += lr * d_thr
        min_thr += 0.1 * lr * d_mn_thr
        max_thr += 0.1 * lr * d_mx_thr

        min_thr = max(0., min_thr)
        max_thr = max(min_thr, max_thr)
        thr = np.clip(thr, min_thr, max_thr)

        if self.lr_activation_threshold > 0.0001:
            self.lr_activation_threshold *= 0.995
        self.activation_threshold = (thr, min_thr, max_thr)
        self.loops += loops

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        w = self.weights if ixs is None else self.weights[ixs]
        return np.sum(np.abs(w), axis=-1)

    def get_relative_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.clip(
            np.abs(np.log2(np.maximum(r, 0.001))),
            0.05, 4.0
        )

    def get_square_root_weights(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        w = self.weights if ixs is None else self.weights[ixs]
        return np.sign(w) * (np.abs(w) ** 0.5)

    def print_stats(self, u, sdr, y):
        sorted_values = np.sort(y)
        ac_size = self.output_sds.active_size
        active_mass = sorted_values[-ac_size:].sum()
        biases = np.log(self.output_rate)
        w = self.weights
        non_zero_mass = np.count_nonzero(np.abs(w) > 0.2 / self.ff_size) / w.size
        thr, mn_thr, mx_thr = self.activation_threshold
        print(
            f'R={self.avg_radius:.3f} E={self.output_entropy():.3f} S={self.output_active_size:.1f}'
            f'| T {thr:.3f} [{mn_thr:.3f}; {mx_thr:.3f}]'
            f'| B [{biases.min():.2f}; {biases.max():.2f}]'
            f'| U {u.min():.3f}  {u.max():.3f}'
            f'| W {w.mean():.3f} [{w.min():.3f}; {w.max():.3f}]  NZ={non_zero_mass:.2f}'
            f'| Y {active_mass:.3f} {sdr.size}'
            f'| {self.loops / self.cnt:.3f}'
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


@numba.jit(nopython=True, cache=True, inline='always')
def get_normal_std(required_r, n_samples: int, p: float) -> float:
    alpha = np.pi / (2 * n_samples)
    alpha = alpha ** (1 / p)
    return required_r * alpha


@numba.jit(nopython=True, cache=True)
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


@numba.jit(nopython=True, cache=True)
def nb_choice_k(max_n, k=1, weights=None, replace=False):
    """
    Choose k samples from max_n values, with optional weights and replacement.

    Args:
        max_n (int): the maximum index to choose
        k (int): number of samples
        weights (array): weight of each index, if not uniform
        replace (bool): whether to sample with replacement
    """
    # Get cumulative weights
    if weights is None:
        weights = np.full(int(max_n), 1.0)
    cumweights = np.cumsum(weights)

    maxweight = cumweights[-1]  # Total of weights
    inds = np.full(k, -1, dtype=np.int64)  # Arrays of sample and sampled indices

    # Sample
    i = 0
    while i < k:

        # Find the index
        r = maxweight * np.random.rand()  # Pick random weight value
        ind = np.searchsorted(cumweights, r, side='right')  # Get corresponding index

        # Optionally sample without replacement
        found = False
        if not replace:
            for j in range(i):
                if inds[j] == ind:
                    found = True
                    continue
        if not found:
            inds[i] = ind
            i += 1

    return inds


@numba.jit(nopython=True, cache=True)
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
