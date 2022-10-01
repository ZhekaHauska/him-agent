#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.modules.htm.temporal_memory import GeneralFeedbackTM
from json import dump
import os
from pathlib import Path


class HTMWriter:
    def __init__(self, name, directory, tm: GeneralFeedbackTM, save_every=None):
        self.directory = directory
        self.name = name
        self.time_step = 0
        self.save_every = save_every
        self.cells = list()
        self.context_symbols = list()
        self.forward_symbols = list()
        self.feedback_symbols = list()

        self.tm = tm
        self.info = {
            'local_range': tm.local_range,
            'context_range': tm.context_range,
            'feedback_range': tm.feedback_range,
            'columns': tm.columns,
            'cells_per_column': tm.cells_per_column,
            'context_cells': tm.context_cells,
            'feedback_cells': tm.feedback_cells,
            'segments_per_cell_apical': tm.max_segments_per_cell_apical,
            'segments_per_cell_basal': tm.max_segments_per_cell_basal,
            'chunk_size': self.save_every
        }

        Path(os.path.join(self.directory, 'stream')).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(directory, name + '_info.json'), 'w') as file:
            dump(self.info, file)

    def write(self, forward_symbol=None, context_symbol=None, feedback_symbol=None, save=False):
        cells = list()
        for cell_id in range(self.tm.total_cells):
            basal_segments = self.tm.basal_connections.segmentsForCell(cell_id)
            apical_segments = self.tm.apical_connections.segmentsForCell(cell_id)
            if self.tm.local_range[0] <= cell_id < self.tm.local_range[1]:
                type_ = 0  # local
                active = cell_id in self.tm.active_cells.sparse
                winner = cell_id in self.tm.winner_cells.sparse
            elif self.tm.context_range[0] <= cell_id < self.tm.context_range[1]:
                type_ = 1  # context
                active = cell_id in self.tm.active_cells_context.sparse
                winner = False
            elif self.tm.feedback_range[0] <= cell_id < self.tm.feedback_range[1]:
                type_ = 2  # feedback
                active = cell_id in self.tm.active_cells_feedback.sparse
                winner = False
            else:
                raise ValueError("Cell id is out of range")

            predictive = cell_id in self.tm.predicted_cells.sparse

            cells.append(
                {
                    'segments': {
                        'basal': self.write_segments(
                            basal_segments,
                            self.tm.active_segments_basal,
                            self.tm.matching_segments_basal,
                            self.tm.basal_connections,
                            self.tm.connected_threshold_basal
                        ),
                        'apical': self.write_segments(
                            apical_segments,
                            self.tm.active_segments_apical,
                            self.tm.matching_segments_apical,
                            self.tm.apical_connections,
                            self.tm.connected_threshold_apical
                        ),
                    },
                    'type': type_,
                    'active': active,
                    'winner': winner,
                    'predictive': predictive,
                    'id': cell_id
                }
            )
        self.cells.append(cells)
        self.forward_symbols.append(forward_symbol)
        self.feedback_symbols.append(feedback_symbol)
        self.context_symbols.append(context_symbol)

        self.time_step += 1

        if save:
            self.save()
        else:
            if self.save_every is not None:
                if (self.time_step % self.save_every) == 0:
                    self.save()

    def save(self):
        with open(os.path.join(self.directory, 'stream', self.name + f'_{self.time_step}.json'), 'w') as file:
            dump(
                {
                    'cells': self.cells,
                    'symbols': {
                        'forward': self.forward_symbols,
                        'feedback': self.feedback_symbols,
                        'context': self.context_symbols
                    }
                }, file
            )

        self.cells.clear()
        self.forward_symbols.clear()
        self.feedback_symbols.clear()
        self.context_symbols.clear()

    @staticmethod
    def write_segments(
            segment_ids, active_segments, matching_segments, connections, permanence_threshold
    ):
        segments = list()
        for segment in segment_ids:
            synapses = connections.synapsesForSegment(segment)
            synapses = [
                (
                    s,
                    connections.presynapticCellForSynapse(s),
                    connections.permanenceForSynapse(s),
                    connections.permanenceForSynapse(s) >= permanence_threshold
                ) for s in synapses]
            if segment in active_segments:
                state = 1  # active
            elif segment in matching_segments:
                state = 2  # matching
            else:
                state = 0  # inactive

            segments.append(
                {
                    'synapses': synapses,
                    'state': state,
                    'id': segment
                }
            )
        return segments
