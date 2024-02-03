#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np

from hima.common.sdrr import AnySparseSdr, split_sdr_values, RateSdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue
from hima.experiments.temporal_pooling.stats.metrics import TMetrics, sdr_similarity


class SdrPredictionTracker:
    sds: Sds
    step_flush_schedule: int | None
    symmetrical_dissimilarity: bool

    miss_rate: MeanValue[float]
    imprecision: MeanValue[float]
    prediction_volume: MeanValue[float]
    dissimilarity: MeanValue[float]

    def __init__(
            self, sds: Sds,
            symmetrical_dissimilarity: bool = True, step_flush_schedule: int = None
    ):
        self.sds = sds
        self.symmetrical_dissimilarity = symmetrical_dissimilarity
        self.step_flush_schedule = step_flush_schedule

        self.dense_cache = np.zeros(sds.size, dtype=float)
        self.predicted_sdr = []
        self.observed_sdr = []

        self.miss_rate = MeanValue()
        self.imprecision = MeanValue()
        self.prediction_volume = MeanValue()
        self.dissimilarity = MeanValue()

    def on_sdr_predicted(self, sdr: AnySparseSdr, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        self.predicted_sdr = sdr
        return {}

    def on_sdr_observed(self, sdr: AnySparseSdr, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        self.observed_sdr = sdr
        return {}

    def on_both_known(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        pr_sdr, pr_value = split_sdr_values(self.predicted_sdr)
        gt_sdr, gt_value = split_sdr_values(self.observed_sdr)

        pr_set_sdr, gt_set_sdr = set(pr_sdr), set(gt_sdr)

        recall = sdr_similarity(pr_set_sdr, gt_set_sdr, symmetrical=False)
        miss_rate = 1 - recall
        self.miss_rate.put(miss_rate)

        precision = sdr_similarity(gt_set_sdr, pr_set_sdr, symmetrical=False)
        imprecision = 1 - precision
        self.imprecision.put(imprecision)

        prediction_volume = safe_divide(len(pr_sdr), self.sds.active_size)
        self.prediction_volume.put(prediction_volume)

        dissimilarity = miss_rate
        if isinstance(self.predicted_sdr, RateSdr):
            similarity = sdr_similarity(
                self.predicted_sdr, RateSdr(sdr=gt_sdr, values=gt_value),
                symmetrical=self.symmetrical_dissimilarity,
                dense_cache=self.dense_cache
            )
            dissimilarity = 1 - similarity
        self.dissimilarity.put(dissimilarity)

        self.predicted_sdr = None
        self.observed_sdr = None

        if self.miss_rate.n_steps == self.step_flush_schedule:
            return self.flush_step_metrics()
        return {}

    def on_sequence_finished(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        if self.step_flush_schedule is None:
            return self.flush_step_metrics()
        return {}

    def flush_step_metrics(self) -> TMetrics:
        if self.miss_rate.n_steps == 0:
            return {}

        miss_rate = self.miss_rate.get()
        imprecision = self.imprecision.get()
        f1_score = safe_divide(
            2 * (1 - miss_rate) * (1 - imprecision),
            (1 - miss_rate) + (1 - imprecision)
        )

        metrics = {
            'f1_score': f1_score,
            'miss_rate': miss_rate,
            'dissimilarity': self.dissimilarity.get(),
            'imprecision': imprecision,
            'prediction_volume': self.prediction_volume.get(),
        }
        self._reset_step_metrics()
        return metrics

    def _reset_step_metrics(self):
        self.miss_rate.reset()
        self.imprecision.reset()
        self.prediction_volume.reset()
        self.dissimilarity.reset()


def get_sdr_prediction_tracker(on: dict, **config) -> SdrPredictionTracker:
    gt_stream = on['sdr_observed']
    return SdrPredictionTracker(sds=gt_stream.sds, **config)
