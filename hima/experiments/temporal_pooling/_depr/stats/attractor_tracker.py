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
from hima.experiments.temporal_pooling._depr.stats.sdr_tracker import SetSdrSequence
from hima.experiments.temporal_pooling._depr.stats.tracker import Tracker, TMetrics


class AttractorTracker(Tracker):
    sds: Sds
    n_repeats_per_step: int | None

    sequence: SetSdrSequence
    step_cum_sym_diff_rate: np.ndarray | list
    i_intra_sequence_step: int
    i_intra_step_repeat: int

    def __init__(self, sds: Sds):
        self.sds = sds
        self.n_repeats_per_step = None
        self._reset_epoch()

    def _reset_epoch(self):
        if self.n_repeats_per_step is None:
            self.step_cum_sym_diff_rate = []
        else:
            self.step_cum_sym_diff_rate = np.zeros(self.n_repeats_per_step)

        self.i_intra_sequence_step = 0
        self._reset_step()

    def _reset_step(self):
        self.sequence = []
        self.i_intra_step_repeat = 0

    def on_sequence_started(self, sequence_id: int):
        self._reset_step()
        self.i_intra_sequence_step += 1

    def on_epoch_started(self):
        self._reset_epoch()

    def on_step(self, sdr: SparseSdr):
        self.i_intra_sequence_step += 1
        self.sequence.append(set(sdr))

        # step metrics
        sdr: set = self.sequence[-1]
        prev_sdr = self.sequence[-2] if len(self.sequence) > 1 else set()
        sym_diff = safe_divide(
            len(sdr ^ prev_sdr),
            len(sdr | prev_sdr)
        )

        if self.n_repeats_per_step is None:
            self.step_cum_sym_diff_rate.append(sym_diff)
        else:
            self.step_cum_sym_diff_rate[self.i_intra_step_repeat] += sym_diff

    def on_step_finished(self):
        if self.n_repeats_per_step is None:
            self.n_repeats_per_step = self.i_intra_step_repeat
            self.step_cum_sym_diff_rate = np.array(self.step_cum_sym_diff_rate)
        self._reset_step()

    def step_metrics(self) -> TMetrics:
        return {}

    def aggregate_metrics(self) -> TMetrics:
        assert isinstance(self.step_cum_sym_diff_rate, np.ndarray)

        avg_sym_diff_rate = self.step_cum_sym_diff_rate / self.i_intra_sequence_step
        agg_metrics = {}
        for i_step in range(1, self.n_repeats_per_step):
            agg_metrics[f'epoch/avg_sym_diff_rate[{i_step}]'] = avg_sym_diff_rate[i_step]

        return agg_metrics
