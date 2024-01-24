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


class SpTracker:
    sp: Any
    step_flush_schedule: int | None

    track_split: bool
    potentials_quantile: float

    potentials: MeanValue[npt.NDArray[float]]
    recognition_strength: MeanValue[float]
    weights: MeanValue[npt.NDArray[float]]

    def __init__(
            self, sp, step_flush_schedule: int = None,
            track_split: bool = False, potentials_quantile: float = 0.5,
            **_
    ):
        self.sp = sp
        self.supported = getattr(sp, 'get_step_debug_info', None) is not None
        self.step_flush_schedule = step_flush_schedule
        self.track_split = track_split

        if not self.supported:
            return
        self.potentials_size = round(potentials_quantile * sp.output_sds.size)
        self.potentials = MeanValue(self.potentials_size)

        self.recognition_strength = MeanValue()
        target_rf_size = round(sp.get_target_rf_sparsity() * sp.feedforward_sds.size)
        self.weights = MeanValue(target_rf_size)

        if self.track_split:
            self.split_size = self.sp.output_sds.size
            self.split_ratio = MeanValue()
            self.split_mass = MeanValue()

    def _reset_aggregate_metrics(self):
        self.potentials.reset()
        self.recognition_strength.reset()
        self.weights.reset()

        if self.track_split:
            self.split_ratio.reset()
            self.split_mass.reset()

    def on_sp_computed(self, _, ignore: bool) -> TMetrics:
        if ignore or not self.supported:
            return {}

        debug_info = self.sp.get_step_debug_info()

        self.potentials.put(debug_info['potentials'][-self.potentials_size:])

        recognition_strength = debug_info.get('recognition_strength')
        self.recognition_strength.put(
            recognition_strength.mean() if len(recognition_strength) > 0 else 0.
        )

        weights = debug_info.get('weights')
        avg_weights = np.sort(weights, axis=1).mean(axis=0)
        avg_weights = avg_weights[-self.weights.agg_value.size:]
        self.weights.put(avg_weights)

        if self.track_split:
            rf = debug_info.get('rf')
            non_recurrent_shift = self.sp.feedforward_sds.size - self.split_size
            mask = rf.flatten() >= non_recurrent_shift
            self.split_ratio.put(np.count_nonzero(mask) / rf.size)
            self.split_mass.put(np.sum(weights.flatten()[mask]) / weights.shape[0])

        if self.potentials.n_steps == self.step_flush_schedule:
            return self.flush_aggregate_metrics()
        return {}

    def on_sequence_finished(self, _, ignore: bool) -> TMetrics:
        if ignore or not self.supported:
            return {}

        if self.step_flush_schedule is None:
            return self.flush_aggregate_metrics()
        return {}

    def flush_aggregate_metrics(self) -> TMetrics:
        if self.potentials.n_steps == 0:
            return {}

        metrics = {
            'potentials': self.potentials.get(),
            'recognition_strength': self.recognition_strength.get(),
            'weights': self.weights.get() * len(self.weights.agg_value)
        }
        if self.track_split:
            metrics['split_ratio'] = self.split_ratio.get()
            metrics['split_mass'] = self.split_mass.get()

        self._reset_aggregate_metrics()
        return metrics


def get_sp_tracker(on: dict, **config) -> SpTracker:
    tracked_stream = on['sp_computed']
    sp = getattr(tracked_stream.owner, 'sp', None)
    if sp is None:
        sp = getattr(tracked_stream.owner, 'tm', None)
    return SpTracker(sp=sp, **config)
