#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config.base import extracted, resolve_absolute_quantity
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.stp.general_feedback_tm import GeneralFeedbackTM


class GeneralFeedbackTemporalMemoryBlock(Block):
    family = 'temporal_memory'

    FEEDFORWARD = 'feedforward'
    ACTIVE_CELLS = 'active_cells'
    PREDICTED_CELLS = 'predicted_cells'
    CORRECTLY_PREDICTED_CELLS = 'correctly_predicted_cells'
    supported_streams = {FEEDFORWARD, ACTIVE_CELLS, PREDICTED_CELLS, CORRECTLY_PREDICTED_CELLS}

    tm: GeneralFeedbackTM

    def align_dimensions(self) -> bool:
        cells_per_column = self._config['cells_per_column']

        propagate_from = None
        for _, stream in self.streams.items():
            if stream.valid:
                propagate_from = stream
                break

        if propagate_from is None:
            return False

        for _, stream in self.streams.items():
            if propagate_from.name == self.FEEDFORWARD and stream.name in [
                self.ACTIVE_CELLS, self.PREDICTED_CELLS, self.CORRECTLY_PREDICTED_CELLS
            ]:
                size = propagate_from.sds.size * cells_per_column
            elif stream.name == self.FEEDFORWARD and propagate_from.name in [
                self.ACTIVE_CELLS, self.PREDICTED_CELLS, self.CORRECTLY_PREDICTED_CELLS
            ]:
                size = propagate_from.sds.size // cells_per_column
            else:
                size = propagate_from.sds.size
            sds = Sds(size=size, active_size=propagate_from.sds.active_size)
            stream.join_sds(sds)
        return True

    def compile(self):
        cells_per_column = self._config['cells_per_column']
        tm_config = self._config
        ff_sds = self.streams[self.FEEDFORWARD].sds
        (
            tm_config, activation_threshold_basal, learning_threshold_basal,
            activation_threshold_apical, learning_threshold_apical,
            max_synapses_per_segment_basal
        ) = extracted(
            tm_config, 'activation_threshold_basal', 'learning_threshold_basal',
            'activation_threshold_apical', 'learning_threshold_apical',
            'max_synapses_per_segment_basal'
        )

        self.tm = GeneralFeedbackTM(
            columns=ff_sds.size,
            context_cells=ff_sds.size * cells_per_column,
            feedback_cells=0,
            activation_threshold_basal=resolve_absolute_quantity(
                activation_threshold_basal, baseline=ff_sds.active_size
            ),
            learning_threshold_basal=resolve_absolute_quantity(
                learning_threshold_basal, baseline=ff_sds.active_size
            ),
            activation_threshold_apical=1,
            learning_threshold_apical=1,
            max_synapses_per_segment_basal=resolve_absolute_quantity(
                max_synapses_per_segment_basal, baseline=ff_sds.active_size
            ),
            **tm_config
        )

    def reset(self):
        self.tm.reset()
        super(GeneralFeedbackTemporalMemoryBlock, self).reset()

    def compute(self, learn: bool = True):
        self.predict(learn)
        self.activate(learn)

    def predict(self, learn: bool = True):
        self.tm.set_active_context_cells(self.streams[self.ACTIVE_CELLS].sdr)
        self.tm.activate_basal_dendrites(learn)
        self.tm.predict_cells()

        if self.PREDICTED_CELLS in self.streams:
            self.streams[self.PREDICTED_CELLS].sdr = self.tm.get_predicted_cells()

    def set_predicted_cells(self):
        self.tm.set_predicted_cells(self.streams[self.PREDICTED_CELLS].sdr)

    def activate(self, learn: bool = True):
        self.tm.set_active_columns(self.streams[self.FEEDFORWARD].sdr)
        self.tm.activate_cells(learn)

        self.streams[self.ACTIVE_CELLS].sdr = self.tm.get_active_cells()
        if self.CORRECTLY_PREDICTED_CELLS in self.streams:
            self.streams[self.CORRECTLY_PREDICTED_CELLS].sdr = self.tm.get_correctly_predicted_cells()
