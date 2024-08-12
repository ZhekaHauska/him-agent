#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hima.common.scheduler import Scheduler
from hima.common.sdr import unwrap_as_rate_sdr, RateSdr, AnySparseSdr
from hima.common.sdr_array import SdrArray
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import entropy, TMetrics


# TODO: standardize LearningRateParam and repeating countdowns setting via config,
#   make reasonable defaults, and add documentation describing the usage. For now,
#   I frequently forget the internal logic of Tracker classes and have to look it up.


class SdrTracker:
    sds: Sds
    step_flush_scheduler: Scheduler
    aggregate_flush_scheduler: Scheduler

    # NB: ..._relative_rate means relative to the expected sds active size

    # current step (=instant) metrics for currently active sdr
    sdr_size: MeanValue[float]
    sym_similarity: MeanValue[float]

    histogram: MeanValue[npt.NDArray[float]]
    union: set[int]

    def __init__(
            self, sds: Sds, step_flush_schedule: int = None, aggregate_flush_schedule: int = None
    ):
        self.sds = sds
        self.step_flush_scheduler = Scheduler(step_flush_schedule)
        self.aggregate_flush_scheduler = Scheduler(aggregate_flush_schedule)

        self.prev_sdr = set()
        fast_lr = LearningRateParam(window=1_000)
        slow_lr = LearningRateParam(window=10_000)

        self.sdr_size = MeanValue(lr=fast_lr)
        self.sym_similarity = MeanValue(lr=fast_lr)
        self.histogram = MeanValue(
            size=self.sds.size, lr=slow_lr, initial_value=self.sds.sparsity
        )
        self.union = set()

    def _reset_step_metrics(self):
        # self.sdr_size.reset(hard=True)
        # self.sym_similarity.reset(hard=True)
        pass

    def _reset_aggregate_metrics(self):
        # self.histogram.reset(hard=True)
        self.union.clear()

    def on_sdr_batch_updated(self, batch: SdrArray, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        # NB: handle sdrs in the batch sequentially, remember the last step and aggregate metrics
        # and return them combined in the end
        step_metrics, agg_metrics = {}, {}
        for i in range(len(batch)):
            if batch.sparse is not None:
                sdr = batch.sparse[i]
            else:
                sdr = np.flatnonzero(batch.dense[i])
                values = batch.dense[i][sdr]
                sdr = RateSdr(sdr, values)

            self._on_sdr_updated(sdr)

            if self.step_flush_scheduler.tick():
                step_metrics = self.flush_step_metrics()
            if self.aggregate_flush_scheduler.tick():
                agg_metrics = self.flush_aggregate_metrics()

        return step_metrics | agg_metrics

    def on_sdr_updated(self, sdr: AnySparseSdr, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        self._on_sdr_updated(sdr)

        metrics = {}
        if self.step_flush_scheduler.tick():
            metrics |= self.flush_step_metrics()
        if self.aggregate_flush_scheduler.tick():
            metrics |= self.flush_aggregate_metrics()
        return metrics

    def _on_sdr_updated(self, sdr: AnySparseSdr):
        list_sdr, value = unwrap_as_rate_sdr(sdr)
        sdr: set = set(list_sdr)
        prev_sdr = self.prev_sdr

        self.histogram.put(value, sdr=list_sdr)

        self.union |= sdr
        sdr_size = len(sdr)
        self.sdr_size.put(sdr_size)

        sym_similarity = safe_divide(len(sdr & prev_sdr), len(sdr | prev_sdr))
        self.sym_similarity.put(sym_similarity)

        self.prev_sdr = sdr

    def on_sequence_finished(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        metrics = {}
        if self.step_flush_scheduler.is_infinite:
            metrics |= self.flush_step_metrics()
        if self.aggregate_flush_scheduler.is_infinite:
            metrics |= self.flush_aggregate_metrics()
        return metrics

    def flush_step_metrics(self) -> TMetrics:
        sdr_size = self.sdr_size.get()
        sparsity = safe_divide(sdr_size, self.sds.size)
        relative_sparsity = safe_divide(sdr_size, self.sds.active_size)

        metrics = {
            'sparsity': sparsity,
            'sparsity_rel': relative_sparsity,
            'similarity': self.sym_similarity.get(),
        }
        self._reset_step_metrics()
        return metrics

    def flush_aggregate_metrics(self) -> TMetrics:
        union_relative_sparsity = safe_divide(len(self.union), self.sds.active_size)
        pmf = self.histogram.get()
        relative_pmf = safe_divide(pmf, pmf.mean())
        log_relative_pmf = np.log(relative_pmf)

        metrics = {
            'H': entropy(pmf),
            'union_sparsity_rel': union_relative_sparsity,
            'PMF_rel': relative_pmf,
            'ln(PMF_rel)': log_relative_pmf,
        }
        self._reset_aggregate_metrics()
        return metrics


def get_sdr_tracker(on: dict = None, **config) -> SdrTracker:
    if on is None:
        return SdrTracker(**config)

    tracked_stream = on['sdr_updated']
    return SdrTracker(sds=tracked_stream.sds, **config)
