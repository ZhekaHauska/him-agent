#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np

from hima.common.config_utils import resolve_init_params, extracted, resolve_absolute_quantity
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.modules.htm.apical_tiebreak_sequence_tm import ApicalTiebreakSequenceMemory


class TemporalMemoryBlockStats:
    recall: float

    def __init__(self):
        self.recall = 0.

    def update(self, active_cells: SparseSdr, correctly_predicted_cells: SparseSdr):
        self.recall = safe_divide(correctly_predicted_cells.size, active_cells.size)

    def step_metrics(self) -> dict[str, Any]:
        return {
            'recall': self.recall
        }

    @staticmethod
    def final_metrics() -> dict[str, Any]:
        # TODO: collect repr and distr
        return {}


class SequenceApicalTemporalMemoryBlock:
    id: int
    name: str
    feedforward_sds: Sds
    cells_per_column: int
    apical_feedback_sds: Sds
    cells_sds: Sds
    output_columns: bool

    tm: ApicalTiebreakSequenceMemory
    tm_config: dict

    stats: TemporalMemoryBlockStats

    _apical_feedback: SparseSdr

    def __init__(self, ff_sds: Sds, output_columns: bool, **partially_resolved_tm_config):
        cells_per_column = partially_resolved_tm_config['cells_per_column']

        self.feedforward_sds = ff_sds
        self.cells_per_column = cells_per_column
        self.cells_sds = Sds(
            size=self.feedforward_sds.size * cells_per_column,
            active_size=self.feedforward_sds.active_size
        )
        self.output_columns = output_columns
        self.tm_config = partially_resolved_tm_config
        self.stats = TemporalMemoryBlockStats()

    @property
    def tag(self) -> str:
        return f'{self.id}_tm'

    @property
    def output_sds(self):
        return self.feedforward_sds if self.output_columns else self.cells_sds

    def configure_apical_feedback(self, fb_sds, resolved_tm_config: dict):
        self.apical_feedback_sds = fb_sds
        self.tm_config = resolved_tm_config
        tm_config = self._to_htm_param_names(**self.tm_config)
        self.tm = ApicalTiebreakSequenceMemory(**tm_config)
        self._apical_feedback = []

    def pass_feedback(self, apical_feedback: SparseSdr):
        self._apical_feedback = apical_feedback

    def reset(self):
        self.tm.reset()
        self._apical_feedback = []

    def reset_stats(self):
        self.stats = TemporalMemoryBlockStats()

    def compute(self, feedforward_input: SparseSdr, learn: bool) -> tuple[SparseSdr, SparseSdr]:
        tm = self.tm

        # if we set empty FB SDS in config, it means we do not want it to be propagated
        apical_feedback = []
        if self.apical_feedback_sds.size > 0:
            apical_feedback = self._apical_feedback

        tm.compute(
            activeColumns=feedforward_input,
            apicalInput=apical_feedback,
            learn=learn
        )

        active_cells = np.array(tm.getActiveCells(), copy=True)
        correctly_predicted_cells = np.array(tm.getPredictedActiveCells(), copy=True)

        self.stats.update(
            active_cells=active_cells,
            correctly_predicted_cells=correctly_predicted_cells
        )

        if self.output_columns:
            active_columns = self._cells_to_columns(active_cells)
            correctly_predicted_columns = self._cells_to_columns(correctly_predicted_cells)
            return active_columns, correctly_predicted_columns

        return active_cells, correctly_predicted_cells

    def _cells_to_columns(self, cells: SparseSdr) -> SparseSdr:
        return np.unique(cells // self.cells_per_column)

    def _to_htm_param_names(
            self, cells_per_column: int, seed: int,
            activation_threshold: float, reduced_basal_threshold: float, learning_threshold: float,
            max_synapses_per_segment: int, max_segments_per_cell: int,
            initial_permanence: float, connected_threshold: float,
            permanence_increment: float, permanence_decrement: float,
            basal_predicted_segment_decrement: float, apical_predicted_segment_decrement: float
    ):
        ff_sds, fb_sds = self.feedforward_sds, self.apical_feedback_sds
        params = dict(
            columnCount=ff_sds.size,
            sampleSize=ff_sds.active_size,
            cellsPerColumn=cells_per_column,
            apicalInputSize=fb_sds.size,

            activationThreshold=activation_threshold,
            reducedBasalThreshold=reduced_basal_threshold,
            minThreshold=learning_threshold,

            maxSynapsesPerSegment=max_synapses_per_segment,
            maxSegmentsPerCell=max_segments_per_cell,

            initialPermanence=initial_permanence,
            connectedPermanence=connected_threshold,
            permanenceIncrement=permanence_increment,
            permanenceDecrement=permanence_decrement,
            basalPredictedSegmentDecrement=basal_predicted_segment_decrement,
            apicalPredictedSegmentDecrement=apical_predicted_segment_decrement,
            seed=seed
        )
        return params


def resolve_tm(tm_config, ff_sds: Sds, seed: int):
    # resolve only what is available already
    tm_config = resolve_init_params(
        tm_config, raise_if_not_resolved=False,
        ff_sds=ff_sds, seed=seed
    )
    tm_config, ff_sds = extracted(tm_config, 'ff_sds')
    # if FF SDS was defined in config, it isn't an Sds object
    ff_sds = Sds.as_sds(ff_sds)

    # resolve connections absolute params (shared for both BC and FB)
    tm_config |= resolve_tm_connections_region(tm_config, ff_sds)

    return SequenceApicalTemporalMemoryBlock(ff_sds=ff_sds, **tm_config)


def resolve_tm_apical_feedback(fb_sds: Sds, tm_block: SequenceApicalTemporalMemoryBlock):
    tm_config = tm_block.tm_config

    # resolve FB SDS setting
    tm_config = resolve_init_params(
        tm_config, raise_if_not_resolved=False,
        fb_sds=fb_sds
    )
    tm_config, fb_sds = extracted(tm_config, 'fb_sds')
    # if FB SDS was defined in config, it's not an Sds object
    fb_sds = Sds.as_sds(fb_sds)

    tm_block.configure_apical_feedback(fb_sds=fb_sds, resolved_tm_config=tm_config)


def resolve_tm_connections_region(connections_config, sds):
    active_size = sds.active_size
    activation_threshold = resolve_absolute_quantity(
        connections_config['activation_threshold'],
        baseline=active_size
    )
    reduced_basal_threshold = resolve_absolute_quantity(
        connections_config['reduced_basal_threshold'],
        baseline=active_size
    )
    learning_threshold = resolve_absolute_quantity(
        connections_config['learning_threshold'],
        baseline=active_size
    )
    max_synapses_per_segment = resolve_absolute_quantity(
        connections_config['max_synapses_per_segment'],
        baseline=active_size
    )
    induced_config = dict(
        activation_threshold=activation_threshold,
        reduced_basal_threshold=reduced_basal_threshold,
        learning_threshold=learning_threshold,
        max_synapses_per_segment=max_synapses_per_segment
    )
    connections_config = connections_config | induced_config
    return connections_config
