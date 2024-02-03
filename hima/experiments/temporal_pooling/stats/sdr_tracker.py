#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hima.common.sdr import SparseSdr
from hima.common.sdrr import split_sdr_values
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue
from hima.experiments.temporal_pooling.stats.metrics import entropy, TMetrics
from hima.experiments.temporal_pooling.stp.sp_utils import (
    RepeatingCountdown,
    make_repeating_counter, tick, is_infinite
)


class SdrTracker:
    sds: Sds
    step_flush_scheduler: RepeatingCountdown
    aggregate_flush_scheduler: RepeatingCountdown

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
        self.step_flush_scheduler = make_repeating_counter(step_flush_schedule)
        self.aggregate_flush_scheduler = make_repeating_counter(aggregate_flush_schedule)

        self.prev_sdr = set()
        self.sdr_size = MeanValue()
        self.sym_similarity = MeanValue()
        self.histogram = MeanValue(size=self.sds.size)
        self.histogram.put(self.sds.sparsity)

        self.union = set()

    def _reset_step_metrics(self):
        self.sdr_size.reset()
        self.sym_similarity.reset()

    def _reset_aggregate_metrics(self):
        self.histogram.reset()
        self.histogram.put(self.sds.sparsity)

        self.union.clear()

    def on_sdr_updated(self, sdr: SparseSdr, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        list_sdr, value = split_sdr_values(sdr)
        sdr: set = set(list_sdr)
        prev_sdr = self.prev_sdr

        self.histogram.put(value, sdr=list_sdr)

        self.union |= sdr
        sdr_size = len(sdr)
        self.sdr_size.put(sdr_size)

        sym_similarity = safe_divide(len(sdr & prev_sdr), len(sdr | prev_sdr))
        self.sym_similarity.put(sym_similarity)

        self.prev_sdr = sdr

        metrics = {}
        flush_step_now, self.step_flush_scheduler = tick(self.step_flush_scheduler)
        if flush_step_now:
            metrics |= self.flush_step_metrics()

        flush_aggregate_now, self.aggregate_flush_scheduler = tick(self.aggregate_flush_scheduler)
        if flush_aggregate_now:
            metrics |= self.flush_aggregate_metrics()

        return metrics

    def on_sequence_finished(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        metrics = {}
        if is_infinite(self.step_flush_scheduler):
            metrics |= self.flush_step_metrics()
        if is_infinite(self.aggregate_flush_scheduler):
            metrics |= self.flush_aggregate_metrics()
        return metrics

    def flush_step_metrics(self) -> TMetrics:
        sdr_size = self.sdr_size.get()
        sparsity = safe_divide(sdr_size, self.sds.size)
        relative_sparsity = safe_divide(sdr_size, self.sds.active_size)

        metrics = {
            'sparsity': sparsity,
            'relative_sparsity': relative_sparsity,
            'sym_similarity': self.sym_similarity.get(),
        }
        self._reset_step_metrics()
        return metrics

    def flush_aggregate_metrics(self) -> TMetrics:
        union_relative_sparsity = safe_divide(len(self.union), self.sds.active_size)
        relative_pmf = safe_divide(self.histogram.get(), self.sds.sparsity)
        log_relative_pmf = np.log(relative_pmf)

        # NB: additional normalization by using avg mass instead of step counts —
        # this way we can compare entropy regardless of the input avg rate
        pmf_for_entropy = self.histogram.get()

        metrics = {
            # TODO: fix entropy calculation with respect to avg mass
            'entropy': entropy(pmf_for_entropy, self.sds),
            'union_relative_sparsity': union_relative_sparsity,
            'relative_pmf': relative_pmf,
            'log_relative_pmf': log_relative_pmf,
        }
        self._reset_aggregate_metrics()
        return metrics


def get_sdr_tracker(on: dict, **config) -> SdrTracker:
    tracked_stream = on['sdr_updated']
    return SdrTracker(sds=tracked_stream.sds, **config)
