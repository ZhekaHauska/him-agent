#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import math

import numpy as np

from hima.experiments.temporal_pooling.stp.sp import SpatialPooler
from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerEnsemble


class SpatialPoolerDecoder:
    def __init__(self, sp: SpatialPooler | SpatialPoolerEnsemble):
        self.sp = sp

    def decode(self, output_probs, learn=False):
        is_ensemble = isinstance(self.sp, SpatialPoolerEnsemble)
        if is_ensemble:
            output_probs = output_probs.reshape(self.sp.n_sp, self.sp.getSingleNumColumns())
            input_probs = np.zeros(self.sp.getNumInputs())
            for sp_i in range(output_probs.shape[0]):
                self.backpropagate_output_probs(self.sp.sps[sp_i], output_probs[sp_i], input_probs)

            n_active_input = self.sp.sps[0].ff_avg_active_size
            n_sp = self.sp.n_sp
        else:
            input_probs = np.zeros(self.sp.ff_size)
            self.backpropagate_output_probs(self.sp, output_probs, input_probs)
            n_active_input = self.sp.ff_avg_active_size
            n_sp = 1

        # input_winners = np.sort(
        #     np.argpartition(-input_probs, n_active_input)[:n_active_input]
        # )
        # input_winners = input_winners[input_probs[input_winners] > 0.001]

        input_probs = np.clip(input_probs / n_sp, 0., 1.)
        # input_probs = np.clip(input_probs, 0., 1.)

        return input_probs

    @staticmethod
    def backpropagate_output_probs(sp, output_probs, input_probs):
        rf, w = sp.rf, sp.weights
        prob_weights = w * np.expand_dims(output_probs, 1)
        # accumulate probabilistic weights onto input vector
        np.add.at(input_probs, rf, prob_weights)
