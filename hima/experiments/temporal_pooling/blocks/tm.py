#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

from hima.common.config.base import extracted, resolve_absolute_quantity, TConfig
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.graph.block import Block

FEEDFORWARD = 'feedforward.sdr'
ACTIVE_CELLS = 'active_cells.sdr'
PREDICTED_CELLS = 'predicted_cells.sdr'
CORRECTLY_PREDICTED_CELLS = 'correctly_predicted_cells.sdr'
WINNER_CELLS = 'winner_cells.sdr'


class TemporalMemoryBlock(Block):
    family = 'temporal_memory'
    supported_streams = {
        FEEDFORWARD, ACTIVE_CELLS, PREDICTED_CELLS, CORRECTLY_PREDICTED_CELLS, WINNER_CELLS
    }

    tm: Any | TConfig

    def __init__(self, tm: TConfig, **kwargs):
        super().__init__(**kwargs)
        self.tm = self.model.config.config_resolver.resolve(tm, config_type=dict)

    def fit_dimensions(self) -> bool:
        # TODO: fix required streams

        cells_per_column = self.tm['cells_per_column']
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

            if propagate_from == FEEDFORWARD and name in [
                ACTIVE_CELLS, PREDICTED_CELLS, CORRECTLY_PREDICTED_CELLS, WINNER_CELLS
            ]:
                size = propagate_from_stream.sds.size * cells_per_column
            elif name == FEEDFORWARD and propagate_from in [
                ACTIVE_CELLS, PREDICTED_CELLS, CORRECTLY_PREDICTED_CELLS, WINNER_CELLS
            ]:
                size = propagate_from_stream.sds.size // cells_per_column
            else:
                size = propagate_from_stream.sds.size
            sds = Sds(size=size, active_size=propagate_from_stream.sds.active_size)
            stream.set_sds(sds)
        return True

    def compile(self):
        cells_per_column = self.tm['cells_per_column']
        ff_sds = self[FEEDFORWARD].sds

        (
            tm_config, activation_threshold_basal, learning_threshold_basal,
            activation_threshold_apical, learning_threshold_apical,
            max_synapses_per_segment_basal
        ) = extracted(
            self.tm, 'activation_threshold_basal', 'learning_threshold_basal',
            'activation_threshold_apical', 'learning_threshold_apical',
            'max_synapses_per_segment_basal'
        )

        self.tm = self.model.config.resolve_object(
            tm_config | dict(
                columns=ff_sds.size,
                context_cells=ff_sds.size * cells_per_column,
                feedback_cells=0,
                activation_threshold_basal=resolve_absolute_quantity(
                    activation_threshold_basal, baseline=ff_sds.active_size
                ),
                learning_threshold_basal=resolve_absolute_quantity(
                    learning_threshold_basal, baseline=ff_sds.active_size
                ),
                activation_threshold_apical=activation_threshold_apical,
                learning_threshold_apical=learning_threshold_apical,
                max_synapses_per_segment_basal=resolve_absolute_quantity(
                    max_synapses_per_segment_basal, baseline=ff_sds.active_size
                ),
            )
        )

    def reset(self):
        self.tm.reset()
        super().reset()

    # =========== API ==========
    def compute(self, learn: bool = True):
        self.predict(learn)
        self.set_predicted_cells()
        self.activate(learn)

    def predict(self, learn: bool = True):
        self.tm.set_active_context_cells(self[ACTIVE_CELLS].get())
        self.tm.activate_basal_dendrites(learn)
        self.tm.predict_cells()

        self[PREDICTED_CELLS].set(self.tm.get_predicted_cells())

    def set_predicted_cells(self):
        self.tm.set_predicted_cells(self[PREDICTED_CELLS].get())

    def union_predicted_cells(self):
        self.tm.union_predicted_cells(self[PREDICTED_CELLS].get())
        self[PREDICTED_CELLS].set(self.tm.get_predicted_cells())

    def set_active_columns(self):
        self.tm.set_active_columns(self[FEEDFORWARD].get())
        self.tm.activate_cells(learn=False)

    def activate(self, learn: bool = True):
        self.tm.set_active_columns(self[FEEDFORWARD].get())
        self.tm.activate_cells(learn)

        self[ACTIVE_CELLS].set(self.tm.get_active_cells())
        self[CORRECTLY_PREDICTED_CELLS].set(self.tm.get_correctly_predicted_cells())
        self[WINNER_CELLS].set(self.tm.get_winner_cells())
