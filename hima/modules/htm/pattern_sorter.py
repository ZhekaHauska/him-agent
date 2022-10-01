#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.modules.htm.connections import Connections

from htm.bindings.sdr import SDR
from htm.bindings.math import Random

import numpy as np

EPS = 1e-12
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"


class PatternToStateMemory:
    def __init__(
            self,
            n_states,
            n_policies,
            min_distance,
            permanence_increment=0.1,
            permanence_decrement=0.01,
            segment_decrement=0.1,
            permanence_connected_threshold=0.5,
            seed=0
    ):
        self.n_states = n_states
        self.n_policies = n_policies
        self.permanence_connected_threshold = permanence_connected_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.segment_decrement = segment_decrement
        self.min_distance = min_distance
        self.learning_threshold = 1 - self.min_distance

        self.connections = Connections(n_states*n_policies, connectedThreshold=self.permanence_connected_threshold,
                                       timeseries=False)

        self._rng = Random(seed)

    def state_to_patterns(self, state, policy):
        cell = self.n_states * policy + state
        cell_segments = self.connections.segmentsForCell(cell)

        patterns = list()
        for segment in cell_segments:
            synapses = self.connections.synapsesForSegment(segment)
            connected = [
                self.connections.presynapticCellForSynapse(syn) for syn in synapses
                if self.connections.permanenceForSynapse(syn) >= self.permanence_connected_threshold
            ]
            patterns.append(connected)
        return patterns

    def connect(self, input_pattern: SDR, state, policy):
        overlap = self.connections.computeActivity(input_pattern, True)
        # state = column, policy = row
        cell = self.n_states * policy + state
        active_cell_segments = np.array(self.connections.segmentsForCell(cell), dtype=UINT_DTYPE)
        # filter segments by cell
        if len(active_cell_segments) > 0:
            overlap = overlap[active_cell_segments]
            num_connected = np.array(
                [self.connections.numConnectedSynapses(seg) for seg in active_cell_segments]
            )
            score = overlap / (num_connected + EPS)

            learning_mask = score > self.learning_threshold
            active_segments = active_cell_segments[learning_mask]
            score = score[learning_mask]
        else:
            score = []
            active_segments = []

        if len(active_segments) > 0:
            active_segment = active_segments[np.argmax(score)]
            self.connections.adaptSegment(active_segment, input_pattern,
                                          self.permanence_increment, self.permanence_decrement)
            max_new = input_pattern.sparse.size
            self.connections.growSynapses(active_segment, input_pattern.sparse,
                                          self.permanence_connected_threshold, self._rng, max_new)
        else:
            new_segment = self.connections.createSegment(cell, 255)
            self.connections.growSynapses(new_segment, input_pattern.sparse, self.permanence_connected_threshold, self._rng,
                                          input_pattern.sparse.size)
