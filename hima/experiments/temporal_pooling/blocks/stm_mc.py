#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

import numpy as np

from hima.common.config.base import TConfig, extracted
from hima.common.config.values import resolve_init_params
from hima.common.sdr import RateSdr, unwrap_as_rate_sdr
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

    def __init__(self, tm: TConfig, learn_during_prediction: bool, **kwargs):
        super().__init__(**kwargs)
        self.learn_during_prediction = learn_during_prediction
        self.use_context = False
        self.tm = self.model.config.config_resolver.resolve(tm, config_type=dict)

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

        sds_map = {
            FEEDFORWARD: self[FEEDFORWARD].sds,
            STATE: self[STATE].sds
        }
        if self.use_context:
            sds_map[CONTEXT] = self[CONTEXT].sds

        required_compartments = {
            to_compartment_name(compartment): sds_map[compartment]
            for compartment in sds_map
        }

        tm_config, tm_factory = self.model.config.resolve_object_requirements(
            self.tm, output_sds=self[STATE].sds, compartments=required_compartments.keys()
        )

        tm_config, compartments_config = extracted(tm_config, 'compartments_config')

        # NB: it iterates over only the required compartments
        tm_config['compartments_config'] = {
            compartment: resolve_init_params(
                # before resolving init params, we need to resolve the config itself
                self.model.config.config_resolver.resolve(
                    compartments_config[compartment], config_type=dict,
                ),
                feedforward_sds=required_compartments[compartment]
            )
            for compartment in required_compartments
        }

        self.tm = tm_factory(**tm_config)

    def reset(self):
        super().reset()

    def prepare_input(self, use_ff, use_context, use_state):
        ff_sdr = self[FEEDFORWARD].get() if use_ff else []
        state_sdr = self[STATE].get() if use_state else []

        compartments_input = {
            to_compartment_name(FEEDFORWARD): ff_sdr,
            to_compartment_name(STATE): state_sdr
        }

        if self.use_context:
            context_sdr = self[CONTEXT].get() if use_context else []
            compartments_input[to_compartment_name(CONTEXT)] = context_sdr

        return compartments_input

    # =========== API ==========
    def reset_ff(self):
        self[FEEDFORWARD].set([], reset=True)
        assert len(self[FEEDFORWARD].get()) == 0

    def compute(self):
        learn = self.model.streams[VARS_LEARN].get()
        compartments_input = self.prepare_input(use_ff=True, use_context=True, use_state=True)

        output_sdr = self.tm.compute(compartments_input, learn=learn)
        self[ACTIVE_CELLS].set(output_sdr)

    def predict(self):
        # learn = self.model.streams[VARS_LEARN].get() and self.learn_during_prediction
        compartments_input = self.prepare_input(use_ff=True, use_context=True, use_state=True)

        output_sdr = self.tm.predict(compartments_input)
        self[PREDICTED_CELLS].set(output_sdr)

    def set_predicted_cells(self):
        pred_sdr, _ = unwrap_as_rate_sdr(self[PREDICTED_CELLS].get())
        if len(pred_sdr) == 0:
            self[PREDICTED_AND_ACTIVE_CELLS].set(self[ACTIVE_CELLS].get())
        else:
            self[PREDICTED_AND_ACTIVE_CELLS].set(self[PREDICTED_CELLS].get())

    def union_predicted_cells(self):
        pred_sdr, pred_values = unwrap_as_rate_sdr(self[PREDICTED_CELLS].get())
        act_sdr, act_values = unwrap_as_rate_sdr(self[ACTIVE_CELLS].get())

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
        pred_sdr, pred_values = unwrap_as_rate_sdr(self[PREDICTED_CELLS].get())
        act_sdr, act_values = unwrap_as_rate_sdr(self[ACTIVE_CELLS].get())

        overlap_sdr = set(pred_sdr) & set(act_sdr)
        mask = np.array([i for i, x in enumerate(pred_sdr) if x in overlap_sdr], dtype=int)
        pred_sdr = pred_sdr[mask]

        if not isinstance(pred_values, float):
            pred_values = pred_values[mask]

        rate_sdr = RateSdr(pred_sdr, pred_values)
        self[CORRECTLY_PREDICTED_CELLS].set(rate_sdr)


def to_compartment_name(name: str) -> str:
    return name[:-4]
