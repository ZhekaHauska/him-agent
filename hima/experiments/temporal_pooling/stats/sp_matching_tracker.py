#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hima.common.utils import isnone
from hima.experiments.temporal_pooling.stats.mc_sp_tracking_aggregator import \
    SpTrackingCompartmentalAggregator
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue
from hima.experiments.temporal_pooling.stats.metrics import TMetrics
from hima.experiments.temporal_pooling.stp.sp_utils import (
    RepeatingCountdown,
    make_repeating_counter, tick, is_infinite
)


class SpMatchingTracker:
    sp: Any
    step_flush_scheduler: RepeatingCountdown

    potentials_quantile: float

    potentials: MeanValue[npt.NDArray[float]]
    recognition_strength: MeanValue[float]

    def __init__(
            self, sp, step_flush_schedule: int = None, potentials_quantile: float = None
    ):
        self.sp = sp
        self.supported = hasattr(sp, 'get_step_debug_info')
        if not self.supported:
            return

        self.step_flush_scheduler = make_repeating_counter(step_flush_schedule)

        potentials_quantile = isnone(potentials_quantile, 3*sp.output_sds.sparsity)
        self.potentials_size = round(potentials_quantile * sp.output_sds.size)
        self.potentials = MeanValue(size=self.potentials_size)

        self.recognition_strength = MeanValue()
        self.target_rf_size = round(sp.get_target_rf_sparsity() * sp.feedforward_sds.size)

    def _reset_aggregate_metrics(self):
        self.potentials.reset()
        self.recognition_strength.reset()

    def on_sp_computed(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        debug_info = self.sp.get_step_debug_info()

        self.potentials.put(debug_info['potentials'][-self.potentials_size:])

        recognition_strength = debug_info.get('recognition_strength')
        self.recognition_strength.put(
            recognition_strength.mean() if len(recognition_strength) > 0 else 0.
        )

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
        assert self.potentials.n_steps != 0

        potentials = self.potentials.get()
        log_potentials = np.log(potentials + 1e-6)
        metrics = {
            'potentials': potentials,
            'ln(potentials)': log_potentials,
            'recognition_strength': self.recognition_strength.get(),
        }

        self._reset_aggregate_metrics()
        return metrics


def get_sp_matching_tracker(on: dict, **config) -> SpMatchingTracker:
    tracked_stream = on['sp_computed']
    sp = getattr(tracked_stream.owner, 'sp', None)
    if sp is None:
        sp = getattr(tracked_stream.owner, 'tm', None)

    if hasattr(sp, 'compartments'):
        # we deal with multi-compartmental TM/SP => track each compartment separately
        # noinspection PyTypeChecker
        return SpTrackingCompartmentalAggregator(sp=sp, tracker_class=SpMatchingTracker, **config)

    return SpMatchingTracker(sp=sp, **config)
