#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stats.sdr_tracker import SetSdrSequence
from hima.experiments.temporal_pooling.stats.tracker import Tracker, TMetrics


class AttractorTracker(Tracker):
    sds: Sds
    n_steps_per_sequence: int | None

    sequence: SetSdrSequence
    step_cum_sym_diff_rate: np.ndarray | list
    i_intra_epoch_sequence: int
    i_intra_sequence_step: int

    def __init__(self, sds: Sds):
        self.sds = sds
        self.n_steps_per_sequence = None
        self._reset_epoch()

    def _reset_epoch(self):
        if self.n_steps_per_sequence is None:
            self.step_cum_sym_diff_rate = []
        else:
            self.step_cum_sym_diff_rate = np.zeros(self.n_steps_per_sequence)

        self.i_intra_epoch_sequence = 0
        self._reset_sequence()

    def _reset_sequence(self):
        self.sequence = []
        self.i_intra_sequence_step = 0

    def on_sequence_started(self, sequence_id: int):
        self._reset_sequence()
        self.i_intra_epoch_sequence += 1

    def on_epoch_started(self):
        self._reset_epoch()

    def on_step(self, sdr: SparseSdr):
        self.sequence.append(set(sdr))

        # step metrics
        sdr: set = self.sequence[-1]
        prev_sdr = self.sequence[-2] if len(self.sequence) > 1 else set()
        sym_diff = safe_divide(
            len(sdr ^ prev_sdr),
            len(sdr | prev_sdr)
        )

        if self.n_steps_per_sequence is None:
            self.step_cum_sym_diff_rate.append(sym_diff)
        else:
            self.step_cum_sym_diff_rate[self.i_intra_sequence_step] += sym_diff
        self.i_intra_sequence_step += 1

    def step_metrics(self) -> TMetrics:
        return {}

    def on_sequence_finished(self):
        if self.n_steps_per_sequence is None:
            self.n_steps_per_sequence = self.i_intra_sequence_step
            self.step_cum_sym_diff_rate = np.array(self.step_cum_sym_diff_rate)

    def aggregate_metrics(self) -> TMetrics:
        assert isinstance(self.step_cum_sym_diff_rate, np.ndarray)

        avg_sym_diff_rate = self.step_cum_sym_diff_rate / self.i_intra_epoch_sequence
        agg_metrics = {}
        for i_step in range(1, self.n_steps_per_sequence):
            agg_metrics[f'epoch/avg_sym_diff_rate[{i_step}]'] = avg_sym_diff_rate[i_step]

        return agg_metrics
