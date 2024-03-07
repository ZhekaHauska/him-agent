#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

import numpy as np

from hima.common.config.base import extracted
from hima.experiments.temporal_pooling.stats.mc_sp_tracking_aggregator import \
    SpTrackingCompartmentalAggregator
from hima.experiments.temporal_pooling.stats.metrics import TMetrics
from hima.experiments.temporal_pooling.stp.sp_utils import (
    RepeatingCountdown,
    make_repeating_counter, tick, is_infinite
)


class SpSynaptogenesisTracker:
    sp: Any
    step_flush_scheduler: RepeatingCountdown

    track_split: bool

    def __init__(self, sp, step_flush_schedule: int = None, track_split: bool = False):
        self.sp = sp
        self.supported = hasattr(sp, 'get_step_debug_info')
        if not self.supported:
            return

        self.step_flush_scheduler = make_repeating_counter(step_flush_schedule)
        self.target_rf_size = round(sp.get_target_rf_sparsity() * sp.feedforward_sds.size)
        self.track_split = track_split
        if self.track_split:
            self.split_size = self.sp.output_sds.size

    def on_sp_computed(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        flush_now, self.step_flush_scheduler = tick(self.step_flush_scheduler)
        if flush_now:
            return self.flush_aggregate_metrics()
        return {}

    def on_sequence_finished(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        if is_infinite(self.step_flush_scheduler):
            return self.flush_aggregate_metrics()
        return {}

    def flush_aggregate_metrics(self) -> TMetrics:
        debug_info = self.sp.get_step_debug_info()

        weights = debug_info.get('weights')
        avg_weights = np.sort(weights, axis=1).mean(axis=0)
        avg_weights = avg_weights[-self.target_rf_size:]

        # expected weight = 1 / target_rf_size => we divide by this normalization term
        normalized_weights = avg_weights * self.target_rf_size
        log_normalized_weights = np.log(normalized_weights)
        metrics = {
            # 'weights': normalized_weights,
            'ln(weights)': log_normalized_weights,
        }

        if self.track_split:
            rf = debug_info.get('rf')
            non_recurrent_shift = self.sp.feedforward_sds.size - self.split_size
            mask = rf.flatten() >= non_recurrent_shift
            split_ratio = np.count_nonzero(mask) / rf.size
            split_mass = np.sum(weights.flatten()[mask]) / weights.shape[0]
            metrics['split_ratio'] = split_ratio
            metrics['split_mass'] = split_mass

        if getattr(self.sp, 'get_health_check_stats', None) is not None:
            metrics |= self.sp.get_health_check_stats(
                self.sp.fast_feedforward_trace,
                self.sp.fast_output_trace,
            )
            metrics['speed_kcps'] = round(1.0 / self.sp.computation_speed.get() / 1000.0, 2)

        return metrics


def get_sp_synaptogenesis_tracker(on: dict, **config) -> SpSynaptogenesisTracker:
    tracked_stream = on['sp_computed']
    sp = getattr(tracked_stream.owner, 'sp', None)
    if sp is None:
        sp = getattr(tracked_stream.owner, 'tm', None)

    if hasattr(sp, 'compartments'):
        # we deal with multi-compartmental TM/SP => track each compartment separately

        # NB: for multi-compartmental SP, there's no need to track splits
        # NB2: always avoid mutating config => create a copy without specific keys
        config, _ = extracted(config, 'track_split')

        # noinspection PyTypeChecker
        return SpTrackingCompartmentalAggregator(
            sp=sp, tracker_class=SpSynaptogenesisTracker, **config
        )

    return SpSynaptogenesisTracker(sp=sp, **config)
