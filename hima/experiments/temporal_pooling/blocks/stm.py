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
from hima.common.sdr import RateSdr, split_sdr_values
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.global_vars import VARS_LEARN

FEEDFORWARD = 'feedforward.sdr'
STATE = 'state.sdr'
CONTEXT = 'context.sdr'
ACTIVE_CELLS = 'active_cells.sdr'
PREDICTED_CELLS = 'predicted_cells.sdr'
CORRECTLY_PREDICTED_CELLS = 'correctly_predicted_cells.sdr'
PREDICTED_AND_ACTIVE_CELLS = 'predicted_and_active_cells.sdr'


class SpatialTemporalMemoryBlock(Block):
    family = 'temporal_memory'
    supported_streams = {
        FEEDFORWARD, CONTEXT, STATE,
        ACTIVE_CELLS, PREDICTED_CELLS, CORRECTLY_PREDICTED_CELLS, PREDICTED_AND_ACTIVE_CELLS
    }

    tm: Any | TConfig

    def __init__(
            self, tm: TConfig, learn_during_prediction: bool,
            forbid_initial_state_connections: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.learn_during_prediction = learn_during_prediction
        self.forbid_initial_state_connections = forbid_initial_state_connections
        self.use_context = False
        self.tm = self.model.config.config_resolver.resolve(tm, config_type=dict)
        self.sdr_concatenator = None

        self.register_stream(PREDICTED_AND_ACTIVE_CELLS)

    def fit_dimensions(self) -> bool:
        active_cells, state = self[ACTIVE_CELLS], self[STATE]
        predicted_cells = self[PREDICTED_CELLS]
        correctly_predicted_cells = self[CORRECTLY_PREDICTED_CELLS]
        predicted_and_active_cells = self[PREDICTED_AND_ACTIVE_CELLS]

        if active_cells.valid_sds:
            state.set_sds(active_cells.sds)
            predicted_cells.set_sds(active_cells.sds)
            correctly_predicted_cells.set_sds(active_cells.sds)
            predicted_and_active_cells.set_sds(active_cells.sds)

        return active_cells.valid_sds

    def compile(self):
        self.use_context = self[CONTEXT] is not None

        sds_list = [self[FEEDFORWARD].sds]
        if self.use_context:
            sds_list.append(self[CONTEXT].sds)
        sds_list.append(self[STATE].sds)

        self.sdr_concatenator = SdrConcatenator(sds_list)
        if self.forbid_initial_state_connections:
            connectable_ff_size = self.sdr_concatenator.output_sds.size - self[STATE].sds.size
        else:
            connectable_ff_size = None

        self.tm = self.model.config.resolve_object(
            self.tm,
            feedforward_sds=self.sdr_concatenator.output_sds,
            output_sds=self[STATE].sds,
            connectable_ff_size=connectable_ff_size
        )

    def reset(self):
        super().reset()

    def prepare_input(self, use_ff, use_context, use_state):
        ff_sdr = self[FEEDFORWARD].get() if use_ff else []
        state_sdr = self[STATE].get() if use_state else []

        if self.use_context:
            context_sdr = self[CONTEXT].get() if use_context else []
            sdrs = [ff_sdr, context_sdr, state_sdr]
        else:
            sdrs = [ff_sdr, state_sdr]

        return self.sdr_concatenator.concatenate(*sdrs)

    # =========== API ==========
    def reset_ff(self):
        self[FEEDFORWARD].set([], reset=True)
        assert len(self[FEEDFORWARD].get()) == 0

    def compute(self):
        learn = self.model.streams[VARS_LEARN].get()
        full_ff_sdr = self.prepare_input(use_ff=True, use_context=True, use_state=True)

        # self.tm.modulation = 0.2
        output_sdr = self.tm.compute(full_ff_sdr, learn=learn)
        # self.tm.modulation = 1.0
        self[ACTIVE_CELLS].set(output_sdr)

    def predict(self):
        learn = self.model.streams[VARS_LEARN].get() and self.learn_during_prediction
        full_ff_sdr = self.prepare_input(use_ff=True, use_context=True, use_state=True)

        output_sdr = self.tm.compute(full_ff_sdr, learn=learn)
        self[PREDICTED_CELLS].set(output_sdr)

    def set_predicted_cells(self):
        pred_sdr, _ = split_sdr_values(self[PREDICTED_CELLS].get())
        if len(pred_sdr) == 0:
            self[PREDICTED_AND_ACTIVE_CELLS].set(self[ACTIVE_CELLS].get())
        else:
            self[PREDICTED_AND_ACTIVE_CELLS].set(self[PREDICTED_CELLS].get())

    def union_predicted_cells(self):
        pred_sdr, pred_values = split_sdr_values(self[PREDICTED_CELLS].get())
        act_sdr, act_values = split_sdr_values(self[ACTIVE_CELLS].get())

        overlap_sdr = set(pred_sdr) & set(act_sdr)
        act_mask = np.array([i for i, x in enumerate(act_sdr) if x not in overlap_sdr], dtype=int)
        act_sdr = act_sdr[act_mask]
        act_values = act_values[act_mask]

        union_sdr = np.concatenate((pred_sdr, act_sdr))
        union_values = np.concatenate((pred_values, act_values))
        indices = np.argsort(union_sdr)
        union_sdr = union_sdr[indices]
        union_values = union_values[indices]

        rate_sdr = RateSdr(union_sdr, union_values)
        self[PREDICTED_AND_ACTIVE_CELLS].set(rate_sdr)

    def set_active_columns(self):
        pass

    def compare_with_prediction(self):
        pred_sdr, pred_values = split_sdr_values(self[PREDICTED_CELLS].get())
        act_sdr, act_values = split_sdr_values(self[ACTIVE_CELLS].get())

        overlap_sdr = set(pred_sdr) & set(act_sdr)
        mask = np.array([i for i, x in enumerate(pred_sdr) if x in overlap_sdr], dtype=int)
        pred_sdr = pred_sdr[mask]
        pred_values = pred_values[mask]

        rate_sdr = RateSdr(pred_sdr, pred_values)
        self[CORRECTLY_PREDICTED_CELLS].set(rate_sdr)
