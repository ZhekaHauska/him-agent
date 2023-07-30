#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np

from hima.experiments.temporal_pooling.stp.sp import SpatialPooler
from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerEnsemble


class SpatialPoolerDecoder:
    def __init__(self, sp: SpatialPooler | SpatialPoolerEnsemble, mode='max'):
        self.sp = sp
        self.mode = mode
        self.receptive_fields = np.zeros((self.sp.getNumColumns(), self.sp.getNumInputs()))

    def decode(self, cell_probs, learn=False):
        assert cell_probs.size == self.sp.getNumColumns()

        is_ensemble = isinstance(self.sp, SpatialPoolerEnsemble)
        if is_ensemble:
            cell_probs = cell_probs.reshape(-1, self.sp.getSingleNumColumns())
            input_probs = np.vstack([
                self.decode_sp(self.sp.sps[sp_i], cell_probs[sp_i])
                for sp_i in range(cell_probs.shape[0])
            ])
            input_probs = input_probs.mean(axis=0)
            n_active_input = self.sp.sps[0].ff_avg_active_size
        else:
            input_probs = self.decode_sp(self.sp, cell_probs)
            n_active_input = self.sp.ff_avg_active_size

        # input_winners = np.sort(
        #     np.argpartition(-input_probs, n_active_input)[:n_active_input]
        # )
        # input_winners = input_winners[input_probs[input_winners] > 0.001]

        sum_probs = input_probs.sum()
        if sum_probs > 0.00001:
            input_probs = np.clip(input_probs / sum_probs, 0., 1.)

        assert input_probs.max() <= 1.0, input_probs.max()
        assert input_probs.min() >= 0., input_probs.min()

        return input_probs

    @staticmethod
    def decode_sp(sp, cell_probs):
        rf, w = sp.rf, sp.weights

        input_probs = np.zeros(sp.ff_size)
        prob_weights = w * np.expand_dims(cell_probs, -1)
        # accumulate probabilistic weights onto input vector
        np.add.at(input_probs, rf, prob_weights)
        return input_probs

    def _update_receptive_fields(self):
        is_ensemble = isinstance(self.sp, SpatialPoolerEnsemble)

        for cell in range(self.sp.getNumColumns()):
            field = np.zeros(self.sp.getNumInputs())
            if is_ensemble:
                sp_id, cell_id = divmod(cell, self.sp.getSingleNumColumns())
                field[
                    self.sp.sps[sp_id].rf[cell_id]
                ] = 1
            else:
                field[self.sp.rf[cell]] = 1

            self.receptive_fields[cell] = field
