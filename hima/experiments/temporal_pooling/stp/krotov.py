#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from hima.common.sdr import SparseSdr, DenseSdr
from hima.common.sdrr import RateSdr, AnySparseSdr, OutputMode, split_sdr_values
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import entropy


class KrotovLayer:
    """
    A competitive network implementation from Krotov-Hopfield.
    Source: Unsupervised learning by competing hidden units
        https://pnas.org/doi/full/10.1073/pnas.1820458116
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
            init_radius: float, lebesgue_p: float, neg_hebb_delta: float, repu_n: float,
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
        self.output_mode = OutputMode.RATE

        self.learning_rate = learning_rate

        self.lebesgue_p = lebesgue_p
        self.neg_hebb_delta = neg_hebb_delta
        self.repu_n = repu_n

        shape = (self.output_size, self.ff_size)
        req_radius = init_radius
        init_std = req_radius * np.sqrt(np.pi / 2 / self.ff_size)
        self.weights = self.rng.normal(loc=0.0, scale=init_std, size=shape)

        self.weights_pow_p = self.get_weight_pow_p()
        self.radius = self.get_radius()

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

        x, w = self.dense_input, self.weights
        p, hb_delta = self.lebesgue_p, self.neg_hebb_delta
        w_p = self.weights_pow_p

        y = np.dot(w_p, x)

        sdr = np.flatnonzero(y > 0)
        values = y[sdr] ** self.repu_n
        values /= values.sum()

        output_sdr = RateSdr(sdr, values)
        self.accept_output(output_sdr, learn=learn)

        if not learn:
            return output_sdr

        lr = self.learning_rate

        k1 = self.output_sds.active_size + 1
        top_k1_ix = np.argpartition(y, -k1)[-k1:]
        top_k1_ix = top_k1_ix[np.argsort(y[top_k1_ix])]

        ixs = np.array([top_k1_ix[-1], top_k1_ix[0]], dtype=int)
        dw = np.array([1.0, -hb_delta])

        _x = np.expand_dims(x, 0)
        _dw = np.expand_dims(dw, -1)

        d_weights = _dw * _x - np.expand_dims(dw * y[ixs], -1) * w[ixs]
        d_weights /= np.abs(d_weights).max() + 1e-30

        self.weights[ixs, :] += lr * d_weights
        self.weights_pow_p[ixs, :] = self.get_weight_pow_p(ixs)
        self.radius[ixs] = self.get_radius(ixs)

        self.cnt += 1
        if self.cnt % 1000 == 0:
            sorted_values = np.sort(values)
            ac_size = self.output_sds.active_size
            active_mass = sorted_values[-ac_size:].sum()

            biases = np.log(self.output_rate)
            print(
                f'{self.avg_radius:.3f} {self.output_active_size:.1f}'
                f'| {biases.mean():.2f} [{biases.min():.2f}; {biases.max():.2f}]'
                f'| {y.min():.3f}  {y.max():.3f}'
                f'| {self.weights.mean():.3f}: {self.weights.min():.3f}  {self.weights.max():.3f}'
                f'| {active_mass:.3f} {values.sum():.3f}  {sdr.size}'
            )

        return output_sdr

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.lebesgue_p
        w = self.weights if ixs is None else self.weights[ixs]
        return np.sum(np.abs(w) ** p, axis=-1) ** (1 / p)

    def get_weight_pow_p(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.lebesgue_p
        w = self.weights if ixs is None else self.weights[ixs]
        return np.sign(w) * (np.abs(w) ** (p - 1))

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
