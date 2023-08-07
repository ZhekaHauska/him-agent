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

    def decode(self, output_probs, learn=False):
        is_ensemble = isinstance(self.sp, SpatialPoolerEnsemble)
        if is_ensemble:
            output_probs = output_probs.reshape(self.sp.n_sp, self.sp.single_output_sds.size)
            input_probs = np.zeros(self.sp.feedforward_sds.size)
            for sp_i in range(output_probs.shape[0]):
                self.backpropagate_output_probs(self.sp.sps[sp_i], output_probs[sp_i], input_probs)

            n_active_input = self.sp.sps[0].ff_avg_active_size
        else:
            input_probs = np.zeros(self.sp.ff_size)
            self.backpropagate_output_probs(self.sp, output_probs, input_probs)
            n_active_input = self.sp.ff_avg_active_size

        input_probs = np.clip(safe_divide(input_probs, output_probs.sum()), 0., 1.)
        return input_probs

    def learn(self, output_probs, correct_obs):
        ...

    @staticmethod
    def backpropagate_output_probs(sp, output_probs, input_probs):
        rf, w = sp.rf, sp.weights
        prob_weights = w * np.expand_dims(output_probs, 1) * sp.ff_avg_active_size
        # accumulate probabilistic weights onto input vector
        np.add.at(input_probs, rf, prob_weights)


class SpatialPoolerLearnedDecoder:
    n_updates: int

    def __init__(self, sp: SpatialPooler | SpatialPoolerEnsemble, hidden_dims: tuple):
        self.sp = sp
        self.rng = np.random.default_rng(self.sp.rng.integers(1_000_000))

        from sklearn.neural_network import MLPClassifier
        self.classifier = MLPClassifier(
            hidden_layer_sizes=hidden_dims,
            activation='relu',
            learning_rate_init=0.004,
            random_state=self.rng.integers(1_000_000),
        )
        self.n_updates = 0
        self.total_updates_required = 100_000
        self.stage_size = 1_000

    def decode(self, output_probs, learn=False):
        input_probs = self.classifier.predict_proba([output_probs]).flatten()
        return input_probs

    def learn(self, output_probs, correct_obs):
        if self.n_updates >= self.total_updates_required:
            return

        classes = None
        if self.n_updates == 0:
            classes = np.arange(self.sp.feedforward_sds.size)

        # every stage update will be approximately every `stage`-th time
        stage = (1 + self.n_updates // self.stage_size) ** 0.7
        if self.rng.random() >= 1. / stage:
            return

        self.n_updates += 1
        self.classifier.partial_fit([output_probs], [correct_obs], classes=classes)


class SpatialPoolerLearnedOverNaiveDecoder(SpatialPoolerDecoder):
    def __init__(self, sp: SpatialPooler | SpatialPoolerEnsemble):
        super().__init__(sp)
        from sklearn.neural_network import MLPClassifier
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(200, 200),
            activation='relu',
            learning_rate_init=0.004,
            random_state=self.sp.rng.integers(1_000_000)
        )
        self.first = True

    def decode(self, output_probs, learn=False):
        input_probs = super().decode(output_probs, learn)
        input_probs = self.classifier.predict_proba([input_probs]).flatten()
        return input_probs

    def learn(self, output_probs, correct_obs):
        input_probs = super().decode(output_probs, learn=False)

        classes = None
        if self.first:
            self.first = False
            classes = np.arange(self.sp.feedforward_sds.size)

        self.classifier.partial_fit([input_probs], [correct_obs], classes=classes)
