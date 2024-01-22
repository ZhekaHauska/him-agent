#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hima.common.sdr import SparseSdr
from hima.common.sdrr import split_sdr_values
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue
from hima.experiments.temporal_pooling.stats.metrics import entropy, TMetrics


class SpTracker:
    sp: Any
    aggregate_flush_schedule: int | None

    potentials: MeanValue[npt.NDArray[float]]
    recognition_strength: MeanValue[float]
    weights: MeanValue[npt.NDArray[float]]

    def __init__(
            self, sp, aggregate_flush_schedule: int = None, **_
    ):
        self.sp = sp
        self.aggregate_flush_schedule = aggregate_flush_schedule

        self.potentials = MeanValue(sp.output_sds.size)
        self.recognition_strength = MeanValue()
        target_rf_size = round(sp.get_target_rf_sparsity() * sp.feedforward_sds.size)
        self.weights = MeanValue(target_rf_size)

    def _reset_aggregate_metrics(self):
        self.potentials.reset()
        self.recognition_strength.reset()
        self.weights.reset()

    def on_sp_computed(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        debug_info = self.sp.get_step_debug_info()

        self.potentials.put(debug_info['potentials'])
        self.recognition_strength.put(debug_info['recognition_strength'])

        weights = debug_info.get('weights')
        self.weights.put(weights[-self.weights.agg_value.size:])

        if self.potentials.n_steps == self.aggregate_flush_schedule:
            return self.flush_aggregate_metrics()
        return {}

    def on_sequence_finished(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        if self.aggregate_flush_schedule is None:
            return self.flush_aggregate_metrics()
        return {}

    def flush_aggregate_metrics(self) -> TMetrics:
        if self.potentials.n_steps == 0:
            return {}

        metrics = {
            'potentials': self.potentials.get(),
            'recognition_strength': self.recognition_strength.get(),
            'weights': self.weights.get()
        }
        self._reset_aggregate_metrics()
        return metrics


def get_sp_tracker(on: dict, **config) -> SpTracker:
    tracked_stream = on['sp_computed']
    return SpTracker(sp=tracked_stream.owner.sp, **config)
