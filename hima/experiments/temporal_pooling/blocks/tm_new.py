#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

import numpy as np

from hima.common.config.base import TConfig
from hima.common.sdr_encoders import SdrConcatenator
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.global_vars import VARS_LEARN

FEEDFORWARD = 'feedforward.sdr'
STATE = 'state.sdr'
CONTEXT = 'context.sdr'
OUTPUT = 'output.sdr'
ACTIVE_CELLS = 'active_cells.sdr'
PREDICTED_CELLS = 'predicted_cells.sdr'
CORRECTLY_PREDICTED_CELLS = 'correctly_predicted_cells.sdr'


class NewTemporalMemoryBlock(Block):
    family = 'temporal_memory'
    supported_streams = {
        FEEDFORWARD, STATE, CONTEXT, OUTPUT,
        ACTIVE_CELLS, PREDICTED_CELLS, CORRECTLY_PREDICTED_CELLS
    }

    tm: Any | TConfig

    def __init__(self, tm: TConfig, learn_during_prediction: bool, **kwargs):
        super().__init__(**kwargs)
        self.learn_during_prediction = learn_during_prediction
        self.tm = self.model.config.config_resolver.resolve(tm, config_type=dict)

    def fit_dimensions(self) -> bool:
        active_cells, state = self[ACTIVE_CELLS], self[STATE]
        predicted_cells = self[PREDICTED_CELLS]
        correctly_predicted_cells = self[CORRECTLY_PREDICTED_CELLS]

        if active_cells.valid_sds:
            state.set_sds(active_cells.sds)
            predicted_cells.set_sds(active_cells.sds)
            correctly_predicted_cells.set_sds(active_cells.sds)

        return active_cells.valid_sds

    def compile(self):
        feedforward_sds, state_sds = self[FEEDFORWARD].sds, self[STATE].sds
        self.sdr_concatenator = SdrConcatenator([feedforward_sds, state_sds])

        self.tm = self.model.config.resolve_object(
            self.tm,
            feedforward_sds=self.sdr_concatenator.output_sds,
            output_sds=state_sds
        )

    def reset(self):
        super().reset()

    # =========== API ==========
    def compute(self):
        learn = self.model.streams[VARS_LEARN].get()
        ff_sdr = self[FEEDFORWARD].get()
        state_sdr = self[STATE].get()
        full_ff_sdr = self.sdr_concatenator.concatenate(ff_sdr, state_sdr)

        output_sdr = self.tm.compute(full_ff_sdr, learn=learn)
        self[ACTIVE_CELLS].set(output_sdr)

    def predict(self):
        learn = self.model.streams[VARS_LEARN].get() and self.learn_during_prediction
        state = self[STATE].get()
        full_ff_sdr = self.sdr_concatenator.concatenate([], state)

        output_sdr = self.tm.compute(full_ff_sdr, learn=learn)
        self[PREDICTED_CELLS].set(output_sdr)

    def set_predicted_cells(self):
        pass

    def union_predicted_cells(self):
        pass

    def set_active_columns(self):
        pass

    def compare_with_prediction(self):
        overlap_sdr = list(
            set(self[PREDICTED_CELLS].get()) & set(self[ACTIVE_CELLS].get())
        )
        overlap_sdr = np.array(overlap_sdr, dtype=int)
        overlap_sdr.sort()
        self[CORRECTLY_PREDICTED_CELLS].set(overlap_sdr)
