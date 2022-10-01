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
from hima.experiments.temporal_pooling.blocks.base_block_stats import BlockStats
from hima.modules.htm.temporal_memory import DelayedFeedbackTM


class TemporalMemoryBlockStats(BlockStats):
    recall: float

    def __init__(self, output_sds: Sds):
        super(TemporalMemoryBlockStats, self).__init__(output_sds=output_sds)
        self.recall = 0.

    def update(self, active_cells: SparseSdr, correctly_predicted_cells: SparseSdr):
        self.recall = safe_divide(correctly_predicted_cells.size, active_cells.size)

    def step_metrics(self) -> dict[str, Any]:
        return {
            'recall': self.recall
        }


class ContextTemporalMemoryBlock:
    id: int
    name: str
    feedforward_sds: Sds
    cells_per_column: int
    basal_context_sds: Sds
    apical_feedback_sds: Sds
    cells_sds: Sds

    tm: DelayedFeedbackTM
    tm_config: dict

    stats: TemporalMemoryBlockStats

    _apical_feedback: SparseSdr

    def __init__(self, ff_sds: Sds, bc_sds: Sds, **partially_resolved_tm_config):
        cells_per_column = partially_resolved_tm_config['cells_per_column']

        self.feedforward_sds = ff_sds
        self.cells_per_column = cells_per_column
        self.basal_context_sds = bc_sds
        self.cells_sds = Sds(
            size=self.feedforward_sds.size * cells_per_column,
            active_size=self.feedforward_sds.active_size
        )
        self.tm_config = partially_resolved_tm_config
        self.stats = TemporalMemoryBlockStats(output_sds=self.output_sds)

    @property
    def tag(self) -> str:
        return f'{self.id}_tm'

    @property
    def output_sds(self):
        return self.cells_sds

    def configure_apical_feedback(self, fb_sds, resolved_tm_config):
        self.apical_feedback_sds = fb_sds
        self.tm_config = resolved_tm_config
        self.tm = DelayedFeedbackTM(**self.tm_config)
        self._apical_feedback = []

    def pass_feedback(self, apical_feedback: SparseSdr):
        self._apical_feedback = apical_feedback

    def reset(self):
        self.tm.reset()
        self._apical_feedback = []

    def reset_stats(self):
        self.stats = TemporalMemoryBlockStats(output_sds=self.output_sds)

    def compute(
            self, feedforward_input: SparseSdr, basal_context: SparseSdr, learn: bool
    ) -> tuple[SparseSdr, SparseSdr]:
        tm = self.tm

        tm.set_active_context_cells(basal_context)
        tm.activate_basal_dendrites(learn)

        tm.set_active_feedback_cells(self._apical_feedback)
        tm.activate_apical_dendrites(learn)
        tm.propagate_feedback()

        tm.predict_cells()

        tm.set_active_columns(feedforward_input)
        tm.activate_cells(learn)

        active_cells = np.array(tm.get_active_cells(), copy=True)
        correctly_predicted_cells = np.array(tm.get_correctly_predicted_cells(), copy=True)

        self.stats.update(
            active_cells=active_cells,
            correctly_predicted_cells=correctly_predicted_cells
        )
        return active_cells, correctly_predicted_cells


def resolve_tm(
        tm_config, ff_sds: Sds, bc_sds: Sds, seed: int
):
    # resolve only what is available already
    tm_config = resolve_init_params(
        tm_config, raise_if_not_resolved=False,
        ff_sds=ff_sds, bc_sds=bc_sds, seed=seed
    )
    tm_config, ff_sds, bc_sds, bc_config = extracted(tm_config, 'ff_sds', 'bc_sds', 'basal_context')
    # if FF/BC SDS were defined in config, they aren't Sds objects
    ff_sds = Sds.as_sds(ff_sds)
    bc_sds = Sds.as_sds(bc_sds)

    # resolve quantities based on FF and BC SDS settings
    tm_config = resolve_init_params(
        tm_config, raise_if_not_resolved=False,
        columns=ff_sds.size, context_cells=bc_sds.size,
    )

    # append extracted and resolved BC config
    tm_config |= resolve_tm_connections_region(bc_config, bc_sds, '_basal')

    return ContextTemporalMemoryBlock(
        ff_sds=ff_sds, bc_sds=bc_sds, **tm_config
    )


def resolve_tm_apical_feedback(fb_sds: Sds, tm_block: ContextTemporalMemoryBlock):
    tm_config = tm_block.tm_config

    # resolve FB SDS setting
    tm_config = resolve_init_params(
        tm_config, raise_if_not_resolved=False,
        fb_sds=fb_sds
    )
    tm_config, fb_sds, fb_config = extracted(tm_config, 'fb_sds', 'apical_feedback')
    # if it was defined in config, it's not an Sds object
    fb_sds = Sds.as_sds(fb_sds)

    # resolve quantities based on FB SDS settings; implicitly asserts all other fields are resolved
    tm_config = resolve_init_params(tm_config, feedback_cells=fb_sds.size)

    # append extracted and resolved FB config
    tm_config |= resolve_tm_connections_region(fb_config, fb_sds, '_apical')

    tm_block.configure_apical_feedback(fb_sds=fb_sds, resolved_tm_config=tm_config)


def resolve_tm_connections_region(connections_config, sds, suffix):
    sample_size = sds.active_size
    activation_threshold = resolve_absolute_quantity(
        connections_config['activation_threshold'],
        baseline=sample_size
    )
    learning_threshold = resolve_absolute_quantity(
        connections_config['learning_threshold'],
        baseline=sample_size
    )
    max_synapses_per_segment = resolve_absolute_quantity(
        connections_config['max_synapses_per_segment'],
        baseline=sample_size
    )
    induced_config = dict(
        sample_size=sample_size,
        activation_threshold=activation_threshold,
        learning_threshold=learning_threshold,
        max_synapses_per_segment=max_synapses_per_segment
    )
    connections_config = connections_config | induced_config
    return {
        f'{k}{suffix}': connections_config[k]
        for k in connections_config
    }
