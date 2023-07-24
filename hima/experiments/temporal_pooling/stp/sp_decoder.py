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

        if learn:
            self._update_receptive_fields()

        if self.mode == 'mean':
            probs_for_bit = self.receptive_fields * cell_probs.reshape((-1, 1))
            probs_for_bit = probs_for_bit.mean(axis=0)
        elif self.mode == 'max':
            probs_for_bit = self.receptive_fields * cell_probs.reshape((-1, 1))
            probs_for_bit = probs_for_bit.max(axis=0)
        elif self.mode == 'sum':
            log_product = np.dot(self.receptive_fields.T, np.log(np.clip(1 - cell_probs, 1e-7, 1)))
            probs_for_bit = 1 - np.exp(log_product)
        else:
            raise ValueError(f'There no such mode: "{self.mode}"!')

        return probs_for_bit

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
