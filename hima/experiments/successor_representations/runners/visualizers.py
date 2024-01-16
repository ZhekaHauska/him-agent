#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hima.modules.belief.cortial_column.layer import Layer
from hima.common.sdr import sparse_to_dense


class BaseVisualizer:
    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class DHTMVisualizer(BaseVisualizer):
    def __init__(self, memory: Layer):
        self.memory = memory
        self.context_messages = None

        # create figures
        self.fig_messages = plt.figure('messages')
        self.messages = self.fig_messages.subplot_mosaic(
            [
                ['external', '.', '.'],
                ['context', 'prediction', 'internal'],
                ['.', 'obs_states_prediction', 'obs_states']
            ],
            height_ratios=[0.25, 1, 0.25]
        )
        self.fig_segments = plt.figure('segments')
        self.segments = self.fig_segments.subplot_mosaic(
            [
                ['.', 'external_fields_of_new', '.'],
                ['total_per_cell', 'context_fields_of_new', 'cells_to_grow_new'],
            ],
            height_ratios=[0.25, 1]
        )

    def step(self):
        # update figures
        self._clear_axes()
        self._update_messages()
        self._update_segments()

        plt.show()

    def reset(self):
        self._clear_axes()
        self.context_messages = self.memory.context_messages.copy()

    def _update_messages(self):
        sns.heatmap(
            self.memory.external_messages.reshape(
                1, -1
            ),
            ax=self.messages['external'],
            cbar=False,
            annot=True
        )
        sns.heatmap(
            self.context_messages.reshape(
                -1, self.memory.cells_per_column
            ).T,
            ax=self.messages['context'],
            annot=True,
            cbar=False
        )
        if self.memory.prediction_cells is not None:
            sns.heatmap(
                self.memory.prediction_cells.reshape(
                    -1, self.memory.cells_per_column
                ).T,
                ax=self.messages['prediction'],
                annot=True,
                cbar=False
            )
        sns.heatmap(
            self.memory.internal_forward_messages.reshape(
                -1, self.memory.cells_per_column
            ).T,
            ax=self.messages['internal'],
            annot=True,
            cbar=False
        )
        sns.heatmap(
            self.memory.prediction_columns.reshape(
                1, -1
            ),
            ax=self.messages['obs_states_prediction'],
            annot=True,
            cbar=False
        )

        sns.heatmap(
            self.memory.observation_messages.reshape(
                1, -1
            ),
            ax=self.messages['obs_states'],
            annot=True,
            cbar=False
        )
        self.context_messages = self.memory.context_messages.copy()

    def _update_segments(self):
        segments_per_cell = self._hist2d_segments_per_cell()
        cells_for_new_segments = self._hist2d_new_segments_cells()
        (
            context_fields_for_new_segments,
            external_fields_for_new_segments
        ) = self._hist2d_new_segments_receptive_fields()

        sns.heatmap(
            segments_per_cell,
            ax=self.segments['total_per_cell'],
            cbar=False,
            annot=True
        )

        sns.heatmap(
            cells_for_new_segments,
            ax=self.segments['cells_to_grow_new'],
            cbar=False,
            annot=True
        )

        sns.heatmap(
            context_fields_for_new_segments,
            ax=self.segments['context_fields_of_new'],
            cbar=False,
            annot=True
        )
        sns.heatmap(
            external_fields_for_new_segments,
            ax=self.segments['external_fields_of_new'],
            cbar=False,
            annot=True
        )

    def _hist2d_segments_per_cell(self):
        # segments in use -> cells -> unique counts -> 2d hist
        cells = self.memory.context_factors.connections.mapSegmentsToCells(
            self.memory.context_factors.segments_in_use
        )

        cells, counts = np.unique(cells, return_counts=True)

        segments_per_cell = np.zeros(self.memory.internal_cells)
        segments_per_cell[cells - self.memory.internal_cells_range[0]] = counts
        return segments_per_cell.reshape(-1, self.memory.cells_per_column).T

    def _hist2d_new_segments_cells(self):
        cells_to_grow_segments = np.zeros(self.memory.internal_cells)
        cells_to_grow_segments[
            self.memory.cells_to_grow_new_context_segments - self.memory.internal_cells_range[0]
        ] = 1

        return cells_to_grow_segments.reshape(-1, self.memory.cells_per_column).T

    def _hist2d_new_segments_receptive_fields(self):
        receptive_fields = self.memory.context_factors.receptive_fields[
            self.memory.new_context_segments
        ]
        cells, counts = np.unique(receptive_fields.flatten(), return_counts=True)

        context_mask = (
                (self.memory.context_cells_range[0] <= cells) &
                (cells < self.memory.context_cells_range[1])
        )
        external_mask = (
                (self.memory.external_cells_range[0] <= cells) &
                (cells < self.memory.external_cells_range[1])
        )

        context_cells_counts = np.zeros(self.memory.context_input_size)
        context_cells_counts[cells[context_mask] - self.memory.context_cells_range[0]] = counts[context_mask]
        context_cells_counts = context_cells_counts.reshape(-1, self.memory.cells_per_column).T

        external_cells_counts = np.zeros(self.memory.external_input_size)
        external_cells_counts[cells[external_mask] - self.memory.external_cells_range[0]] = counts[external_mask]
        external_cells_counts = external_cells_counts.reshape(1, -1)
        return context_cells_counts, external_cells_counts

    def _clear_axes(self):
        for title, ax in self.messages.items():
            ax.clear()

        for title, ax in self.segments.items():
            ax.clear()

