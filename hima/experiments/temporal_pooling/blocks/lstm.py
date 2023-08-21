#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

import numpy as np

from hima.common.config.base import TConfig
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.graph.block import Block

FEEDFORWARD = 'feedforward.sdr'
ACTIVE_CELLS = 'active_cells.sdr'
PREDICTED_CELLS = 'predicted_cells.sdr'
CORRECTLY_PREDICTED_CELLS = 'correctly_predicted_cells.sdr'
WINNER_CELLS = 'winner_cells.sdr'


class LstmBlock(Block):
    family = 'temporal_memory'
    supported_streams = {
        FEEDFORWARD, ACTIVE_CELLS, PREDICTED_CELLS, CORRECTLY_PREDICTED_CELLS, WINNER_CELLS
    }

    tm: Any | TConfig

    def __init__(self, tm: TConfig, **kwargs):
        super().__init__(**kwargs)
        self.tm = self.model.config.config_resolver.resolve(tm, config_type=dict)

    def fit_dimensions(self) -> bool:
        required_streams = {
            FEEDFORWARD, ACTIVE_CELLS, PREDICTED_CELLS, CORRECTLY_PREDICTED_CELLS, WINNER_CELLS
        }

        propagate_from, propagate_from_stream = None, None
        for short_name in self.supported_streams:
            stream = self[short_name]
            if stream is not None and stream.valid_sds:
                propagate_from, propagate_from_stream = short_name, stream
                break

        if propagate_from is None:
            return False

        for name in self.supported_streams:
            stream = self[name]
            if stream is None and name in required_streams:
                stream = self.model.register_stream(self.supported_streams[name])
            if stream is None:
                continue

            size = propagate_from_stream.sds.size
            sds = Sds(size=size, active_size=propagate_from_stream.sds.active_size)
            stream.set_sds(sds)
        return True

    def compile(self):
        ff_sds = self[FEEDFORWARD].sds
        self.tm = self.model.config.resolve_object(self.tm, input_size=ff_sds.size)

    def reset(self):
        self.tm.reset()
        super().reset()

    # =========== API ==========
    def compute(self, learn: bool = True):
        self.predict(learn)
        self.set_predicted_cells()
        self.activate(learn)

    def predict(self, learn: bool = True):
        pass

    def set_predicted_cells(self):
        pass

    def union_predicted_cells(self):
        pass

    def set_active_columns(self):
        pass

    def activate(self, learn: bool = True):
        prediction_dense, correctly_predicted_sdr = self.tm.observe(
            self[FEEDFORWARD].get(), learn=learn
        )
        predicted_sdr = np.flatnonzero(prediction_dense > .5)

        self[ACTIVE_CELLS].set([])
        self[CORRECTLY_PREDICTED_CELLS].set(correctly_predicted_sdr)
        self[WINNER_CELLS].set([])
        self[PREDICTED_CELLS].set(predicted_sdr)
