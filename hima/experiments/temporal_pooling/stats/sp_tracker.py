#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hima.experiments.temporal_pooling.stats.mean_value import MeanValue
from hima.experiments.temporal_pooling.stats.metrics import TMetrics
from hima.experiments.temporal_pooling.stp.sp_utils import (
    RepeatingCountdown,
    make_repeating_counter, tick, is_infinite
)


class SpTracker:
    sp: Any
    step_flush_scheduler: RepeatingCountdown

    track_split: bool
    potentials_quantile: float

    weights: MeanValue[npt.NDArray[float]]

    def __init__(self, sp, step_flush_schedule: int = None, track_split: bool = False):
        self.sp = sp
        self.supported = getattr(sp, 'get_step_debug_info', None) is not None
        if not self.supported:
            return

        self.step_flush_scheduler = make_repeating_counter(step_flush_schedule)
        self.track_split = track_split

        self.target_rf_size = round(sp.get_target_rf_sparsity() * sp.feedforward_sds.size)
        self.weights = MeanValue(size=self.target_rf_size)

        if self.track_split:
            self.split_size = self.sp.output_sds.size
            self.split_ratio = MeanValue()
            self.split_mass = MeanValue()

    def _reset_aggregate_metrics(self):
        self.weights.reset()

        if self.track_split:
            self.split_ratio.reset()
            self.split_mass.reset()

    def on_sp_computed(self, _, ignore: bool) -> TMetrics:
        if ignore or not self.supported:
            return {}

        debug_info = self.sp.get_step_debug_info()

        weights = debug_info.get('weights')
        if weights.ndim == 2:
            avg_weights = np.sort(weights, axis=1).mean(axis=0)
            avg_weights = avg_weights[-self.target_rf_size:]
            self.weights.put(avg_weights)

        if self.track_split:
            rf = debug_info.get('rf')
            non_recurrent_shift = self.sp.feedforward_sds.size - self.split_size
            mask = rf.flatten() >= non_recurrent_shift
            self.split_ratio.put(np.count_nonzero(mask) / rf.size)
            self.split_mass.put(np.sum(weights.flatten()[mask]) / weights.shape[0])

        flush_now, self.step_flush_scheduler = tick(self.step_flush_scheduler)
        if flush_now:
            return self.flush_aggregate_metrics()
        return {}

    def on_sequence_finished(self, _, ignore: bool) -> TMetrics:
        if ignore or not self.supported:
            return {}

        if is_infinite(self.step_flush_scheduler):
            return self.flush_aggregate_metrics()
        return {}

    def flush_aggregate_metrics(self) -> TMetrics:
        if self.weights.n_steps == 0:
            return {}

        # expected weight = 1 / target_rf_size => we divide by this normalization term
        normalized_weights = self.weights.get() * self.target_rf_size
        metrics = {
            'weights': normalized_weights,
        }
        if self.track_split:
            metrics['split_ratio'] = self.split_ratio.get()
            metrics['split_mass'] = self.split_mass.get()

        if getattr(self.sp, 'health_check', None) is not None:
            self.sp.health_check()
            metrics |= self.sp.health_check_results
            metrics['speed_kcps'] = round(1.0 / self.sp.computation_speed.get() / 1000.0, 2)

        self._reset_aggregate_metrics()
        return metrics


def get_sp_tracker(on: dict, **config) -> SpTracker:
    tracked_stream = on['sp_computed']
    sp = getattr(tracked_stream.owner, 'sp', None)
    if sp is None:
        sp = getattr(tracked_stream.owner, 'tm', None)
    return SpTracker(sp=sp, **config)
