#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np

from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling._depr.stats.tracker import Tracker, TMetrics

SdrSequence = list[SparseSdr]
SetSdrSequence = list[set[int]]
SeqHistogram = np.ndarray


class AnomalyTracker(Tracker):
    anomaly: float

    def __init__(self):
        self.anomaly = 0.

    def _reset(self):
        self.sequence = []

    def on_sequence_started(self, sequence_id: int):
        self._reset()

    def on_step(self, anomaly: float):
        assert isinstance(anomaly, float)
        self.anomaly = anomaly

    def step_metrics(self) -> TMetrics:
        return {
            'step/anomaly': self.anomaly,
        }
