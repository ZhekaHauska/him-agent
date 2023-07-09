#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional, Union

import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import isnone


class TemporalPooler:
    initial_pooling = 1.0

    sds: Sds
    reset_on_activation: bool
    decay: float
    decay_mx: Optional[np.ndarray]
    activation_threshold: float
    not_predicted_weight: float

    traces: np.ndarray
    rng: Generator

    def __init__(
            self, sds: Sds, sparsity: Union[int, float], seed: int,
            reset_on_activation: bool,
            pooling_window: float = None, decay: float = None, activation_threshold: float = None,
            rand_decay_max_ratio: float = 1.,
            not_predicted_weight: float = .7,
    ):
        # apply configured output sparsity to the input size to get sds of an output
        self.sds = Sds.make([sds.size, sparsity])
        self.rng = np.random.default_rng(seed)

        # no reset:
        #   a) all: threshold, window
        #   b) top k: decay or threshold
        # reset:
        #   a) makes no sense
        #   b) top k: decay or threshold
        self.reset_on_activation = reset_on_activation
        if self.reset_on_activation:
            assert self.sds.sparsity < 1., 'activate all with reset makes no sense!'

        decay, activation_threshold = _resolve_decay_params(
            decay=decay, window=pooling_window, threshold=activation_threshold
        )
        self.decay = decay
        self.activation_threshold = activation_threshold

        self.decay_mx = None
        assert rand_decay_max_ratio >= 1., 'decay_mx_randomization_scale must be >= 1.0'
        if rand_decay_max_ratio > 1.:
            self.decay_mx = _make_decay_matrix(
                rng=self.rng, size=sds.size,
                decay_mean=self.decay, decay_max_ratio=rand_decay_max_ratio
            )

        self.not_predicted_weight = not_predicted_weight
        self.traces = np.zeros(self.sds.size)

    def _apply_decay(self):
        if self.decay_mx is not None:
            decay_mx = self.decay_mx
            self.traces *= decay_mx
        else:
            self.traces *= self.decay

        # FIXME: check if clipping is needed
        # np.clip(self._pooling_activations, 0, 1, out=self._pooling_activations)

    def compute(
            self, feedforward: SparseSdr, predicted_feedforward: SparseSdr = None
    ) -> SparseSdr:
        self._apply_decay()

        init_pooling = self.initial_pooling
        if predicted_feedforward:
            self.traces[feedforward] += init_pooling * self.not_predicted_weight
            self.traces[predicted_feedforward] += init_pooling * (1. - self.not_predicted_weight)
        else:
            self.traces[feedforward] += init_pooling

        # calculate what should be propagated up from the pooling to the Upper SP
        if self.sds.sparsity == 1.:
            # propagate all
            threshold = self.activation_threshold
        else:
            # adapt activation threshold to return not more than top k
            top_k = self.sds.active_size
            threshold = np.partition(self.traces, kth=-top_k)[-top_k]
            # take into account global lower bound
            threshold = max(threshold, self.activation_threshold)

        result = np.flatnonzero(self.traces >= threshold)
        if len(result) > self.sds.active_size:
            result = self.rng.choice(result, size=self.sds.active_size, replace=False)
            result.sort()

        if self.reset_on_activation:
            self.traces[result] = .0
        return result

    def reset(self):
        self.traces.fill(.0)


def _resolve_decay_params(
        decay: float = None, window: float = None, threshold: float = None
) -> tuple[float, float]:
    if decay is not None:
        threshold = threshold if window is None else np.power(decay, window)
    else:
        decay = np.power(threshold, 1 / window)

    threshold = max(isnone(threshold, 0.), 1e-3)
    return decay, threshold


def _make_decay_matrix(rng, size, decay_mean, decay_max_ratio):
    return decay_mean * _loguniform(rng, decay_max_ratio, size=size)


def _loguniform(rng, std, size):
    low = np.log(1 / std)
    high = np.log(std)
    return np.exp(rng.uniform(low, high, size))
