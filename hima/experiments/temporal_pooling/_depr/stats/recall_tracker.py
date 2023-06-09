#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling._depr.stats.tracker import Tracker, TMetrics

SdrSequence = list[SparseSdr]
SetSdrSequence = list[set[int]]
SeqHistogram = np.ndarray


class AnomalyTracker(Tracker):
    n_steps: int
    cum_anomaly: float
    anomaly: float

    def __init__(self):
        self.anomaly = 0.
        self._reset()

    def _reset(self):
        self.cum_anomaly = 0.
        self.n_steps = 0

    def on_epoch_started(self):
        self._reset()

    def on_step(self, anomaly: float):
        assert isinstance(anomaly, float)
        self.anomaly = anomaly
        self.cum_anomaly += anomaly
        self.n_steps += 1

    def step_metrics(self) -> TMetrics:
        return {
            'step/anomaly': self.anomaly,
        }

    def aggregate_metrics(self) -> TMetrics:
        return {
            'agg/anomaly': safe_divide(self.cum_anomaly, self.n_steps)
        }
