#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from hima.common.config.base import TConfig
from hima.common.sdr import (
    SparseSdr, DenseSdr, OutputMode
)
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.stp.pruning_controller_dense import PruningController
from hima.experiments.temporal_pooling.stp.se_general import (
    LearningPolicy, WeightsDistribution
)
from hima.experiments.temporal_pooling.stp.sp_utils import tick
from hima.modules.htm.utils import abs_or_relative


class SpatialEncoderDenseBackend:
    """
    A dense weights for the spatial encoder.
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
    pruned_mask: npt.NDArray[bool] | None
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

            lebesgue_p: float = 1.0, init_radius: float = 10.0,
            weights_distribution: WeightsDistribution = WeightsDistribution.NORMAL,
            inhibitory_ratio: float = 0.0,

            match_p: float = 1.0,

            learning_policy: LearningPolicy = LearningPolicy.LINEAR, persistent_signs: bool = True,
            normalize_dw: bool = False,

            pruning: TConfig = None,
            **kwargs
    ):
        print(f'kwargs: {kwargs}')
        self.rng = np.random.default_rng(seed)

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.output_sds = Sds.make(output_sds)

        # ==> Weights initialization
        n_out, n_in = self.output_sds.size, self.feedforward_sds.size
        w_shape = (n_out, n_in)

        self.lebesgue_p = lebesgue_p
        self.rf_sparsity = 1.0
        self.weights = sample_weights(
            self.rng, w_shape, weights_distribution, init_radius, self.lebesgue_p
        )

        self.radius = self.get_radius()
        self.pos_log_radius = self.get_pos_log_radius()

        # make a portion of weights negative
        if inhibitory_ratio > 0.0:
            inh_mask = self.rng.binomial(1, inhibitory_ratio, size=w_shape).astype(bool)
            self.weights[inh_mask] *= -1.0

        # ==> Pattern matching
        self.match_p = match_p

        # ==> Learning
        self.learning_policy = learning_policy
        # global learn flag that is switched each compute, to avoid passing it through
        # the whole chain on demand. After compute it's set to False automatically
        self.learn = False
        self.normalize_dw = normalize_dw
        self.persistent_signs = persistent_signs

        # ==> Output
        self.pruning_controller = None
        if pruning is not None:
            self.pruning_controller = PruningController(self, **pruning)
            self.pruned_mask = np.zeros_like(self.weights, dtype=bool)

        print(
            f'Init SE backend: {self.lebesgue_p}-norm | {self.avg_radius:.3f}'
            f' | {self.learning_policy} | match W^{self.match_p}'
        )

    def match_input(self, x):
        w, p = self.weights, self.match_p
        if p != 1.0:
            w = np.sign(w) * (np.abs(w) ** p)
        return np.dot(w, x)

    def update_dense_weights(self, x, sdr, y, u, lr):
        if sdr.size == 0:
            return

        w = self.weights[sdr]

        _x = np.expand_dims(x, 0)
        y_ = np.expand_dims(y, -1)
        lr_ = np.expand_dims(lr, -1)

        sg = np.sign(w) if self.persistent_signs else 1.0
        if self.learning_policy == LearningPolicy.KROTOV:
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
        self.pos_log_radius[sdr] = self.get_pos_log_radius(sdr)

    def prune_newborns(self):
        pc = self.pruning_controller
        if not pc.is_newborn_phase:
            return
        now, pc.countdown = tick(pc.countdown)
        if not now:
            return

        from hima.common.timer import timed
        (sparsity, rf_size), t = timed(pc.shrink_receptive_field)(self.pruned_mask)
        stage = pc.stage
        sparsity_pcnt = round(100.0 * sparsity, 1)
        t = round(t * 1000.0, 2)
        print(f'Prune #{stage}: {sparsity_pcnt:.1f}% | {rf_size} | {t} ms')

        self.rf_sparsity = sparsity
        # rescale weights to keep the same norms
        old_radius, new_radius = self.radius, self.get_radius()
        self.weights *= np.expand_dims(old_radius / new_radius, -1)

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.lebesgue_p
        w = self.weights if ixs is None else self.weights[ixs]
        if p == 1:
            # shortcut to remove unnecessary calculations
            return np.sum(np.abs(w), axis=-1)
        return np.sum(np.abs(w) ** p, axis=-1) ** (1 / p)

    def get_pos_log_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.log2(np.maximum(r, 1.0))

    def plot_weights_distr(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        w, r = self.weights, self.radius
        w = w / np.expand_dims(r, -1)
        sns.histplot(w.flatten())
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


def sample_weights(rng, w_shape, distribution, radius, lebesgue_p):
    if distribution == WeightsDistribution.NORMAL:
        init_std = get_normal_std(w_shape[1], lebesgue_p, radius)
        weights = np.abs(rng.normal(loc=0., scale=init_std, size=w_shape))

    elif distribution == WeightsDistribution.UNIFORM:
        init_std = get_uniform_std(w_shape[1], lebesgue_p, radius)
        weights = rng.uniform(0., init_std, size=w_shape)

    else:
        raise ValueError(f'Unsupported distribution: {distribution}')

    return weights


def get_uniform_std(n_samples, p, required_r) -> float:
    alpha = 2 / n_samples
    alpha = alpha ** (1 / p)
    return required_r * alpha


def get_normal_std(n_samples: int, p: float, required_r) -> float:
    alpha = np.pi / (2 * n_samples)
    alpha = alpha ** (1 / p)
    return required_r * alpha


def arg_top_k(x, k):
    k = min(k, x.size)
    return np.argpartition(x, -k, axis=-1)[..., -k:]


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
