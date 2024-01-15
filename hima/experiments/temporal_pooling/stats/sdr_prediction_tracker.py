#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np

from hima.common.sdrr import AnySparseSdr, split_sdr_values, RateSdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stats.metrics import TMetrics, sdr_similarity
from hima.experiments.temporal_pooling.stp.temporal_memory import TemporalMemory


class SdrPredictionTracker:
    tm: TemporalMemory

    def __init__(self, sds: Sds, symmetrical_dissimilarity: bool = True):
        self.sds = sds
        self.symmetrical_dissimilarity = symmetrical_dissimilarity
        self.dense_cache = np.zeros(sds.size, dtype=float)
        self.predicted_sdr = []
        self._reset()

    def _reset(self):
        self.predicted_sdr = []

    def on_sequence_started(self, *_, **__) -> TMetrics:
        self._reset()
        return {}

    def on_sdr_predicted(self, sdr: AnySparseSdr, reset: bool) -> TMetrics:
        if reset:
            self._reset()
            return {}
        self.predicted_sdr = sdr
        return {}

    def on_sdr_observed(self, sdr: AnySparseSdr, reset: bool) -> TMetrics:
        if reset:
            self._reset()
            return {}

        pr_sdr, pr_value = split_sdr_values(self.predicted_sdr)
        gt_sdr, gt_value = split_sdr_values(sdr)

        pr_set_sdr, gt_set_sdr = set(pr_sdr), set(gt_sdr)

        recall = sdr_similarity(pr_set_sdr, gt_set_sdr, symmetrical=False)
        miss_rate = 1 - recall
        precision = sdr_similarity(gt_set_sdr, pr_set_sdr, symmetrical=False)
        imprecision = 1 - precision

        prediction_volume = safe_divide(len(pr_sdr), self.sds.active_size)

        dissimilarity = miss_rate
        if isinstance(self.predicted_sdr, RateSdr):
            dissimilarity = sdr_similarity(
                self.predicted_sdr, RateSdr(sdr=gt_sdr, values=gt_value),
                symmetrical=self.symmetrical_dissimilarity,
                dense_cache=self.dense_cache
            )

        return {
            'prediction_volume': prediction_volume,
            'miss_rate': miss_rate,
            'imprecision': imprecision,
            'dissimilarity': dissimilarity
        }

    def on_sequence_finished(self, _, reset: bool) -> TMetrics:
        if reset:
            return {}
        return {}


def get_sdr_prediction_tracker(on: dict) -> SdrPredictionTracker:
    gt_stream = on['sdr_observed']
    return SdrPredictionTracker(sds=gt_stream.sds)
