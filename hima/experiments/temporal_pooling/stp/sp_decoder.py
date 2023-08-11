#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np

from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stp.sp import SpatialPooler
from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerEnsemble


class SpatialPoolerDecoder:
    def __init__(self, sp: SpatialPooler | SpatialPoolerEnsemble):
        self.sp = sp

    def decode(self, output_probs, learn=False, correct_obs=None):
        is_ensemble = isinstance(self.sp, SpatialPoolerEnsemble)
        if is_ensemble:
            output_probs = output_probs.reshape(self.sp.n_sp, self.sp.single_output_sds.size)
            input_probs = np.zeros(self.sp.feedforward_sds.size)
            for sp_i in range(output_probs.shape[0]):
                self.backpropagate_output_probs(self.sp.sps[sp_i], output_probs[sp_i], input_probs)
        else:
            input_probs = np.zeros(self.sp.ff_size)
            self.backpropagate_output_probs(self.sp, output_probs, input_probs)

        input_probs = np.clip(safe_divide(input_probs, output_probs.sum()), 0., 1.)
        return input_probs

    def learn(self, output_probs, correct_obs, decoded_obs=None):
        ...

    @staticmethod
    def backpropagate_output_probs(sp, output_probs, input_probs):
        rf, w = sp.rf, sp.weights
        prob_weights = w * np.expand_dims(output_probs, 1) * sp.ff_avg_active_size
        # accumulate probabilistic weights onto input vector
        np.add.at(input_probs, rf, prob_weights)


class SpatialPoolerLearnedDecoder:
    n_updates: int

    def __init__(self, sp: SpatialPooler | SpatialPoolerEnsemble, hidden_dims=None):
        self.sp = sp
        self.rng = np.random.default_rng(self.sp.rng.integers(1_000_000))

        shape = (
            self.sp.feedforward_sds.size,
            self.sp.output_sds.size
        )
        self.weights = self.rng.normal(0., 0.01, size=shape)
        self.lr = .25
        self.power_t = 0.05

        self.n_updates = 0
        self.total_updates_required = 100_000
        self.stage_size = 4_000

    def decode(self, output_probs, learn=False, correct_obs=None):
        input_probs = self.weights @ output_probs

        if learn and correct_obs is not None:
            self.learn(
                output_probs=output_probs,
                correct_obs=correct_obs,
                decoded_obs=input_probs
            )

        return input_probs

    def learn(self, output_probs, correct_obs, decoded_obs=None):
        if self.n_updates >= self.total_updates_required:
            return

        # every stage update will be approximately every `stage`-th time
        stage = (1 + self.n_updates // self.stage_size) ** 0.7
        if self.rng.random() >= 1. / stage:
            return

        self.n_updates += 1

        if decoded_obs is None:
            decoded_obs = self.decode(output_probs, learn=False)

        # sigmoid_derivative = decoded_obs * (1 - decoded_obs) * self.scale
        loss_derivative = decoded_obs - correct_obs
        lr = self.lr / self.n_updates ** self.power_t

        self.weights -= np.outer(loss_derivative, lr * output_probs)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))
