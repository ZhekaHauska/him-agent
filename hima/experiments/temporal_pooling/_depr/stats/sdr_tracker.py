#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling._depr.stats.metrics import entropy
from hima.experiments.temporal_pooling._depr.stats.tracker import Tracker, TMetrics

SdrSequence = list[SparseSdr]
SetSdrSequence = list[set[int]]
SeqHistogram = np.ndarray


class SdrTracker(Tracker):
    sds: Sds

    # NB: ..._relative_rate means relative to the expected sds active size

    # current step (=instant) metrics for currently active sdr
    sparsity: float
    relative_sparsity: float
    diff_relative_rate: float
    sym_diff_rate: float
    union_coverage: float
    pmf_coverage: float
    entropy_coverage: float

    # aggregate cluster/sequence/set data or metrics
    sequence: SetSdrSequence
    pmf_coverage_history: list[float]
    aggregate_histogram: np.ndarray
    aggregate_sparsity: float
    aggregate_relative_sparsity: float
    aggregate_entropy: float

    def __init__(self, sds: Sds):
        self.sds = sds
        self._reset()

    def _reset(self):
        self.sequence = []
        self.pmf_coverage_history = []
        self.aggregate_histogram = np.zeros(self.sds.size)

    def aggregate_pmf(self) -> np.ndarray:
        return safe_divide(self.aggregate_histogram, len(self.sequence))

    def on_sequence_started(self, sequence_id: int):
        self._reset()

    def on_step(self, sdr: SparseSdr):
        list_sdr = sdr
        self.sequence.append(set(sdr))
        self.aggregate_histogram[list_sdr] += 1

        # step metrics
        sdr: set = self.sequence[-1]
        sdr_size = len(sdr)
        prev_sdr = self.sequence[-2] if len(self.sequence) > 1 else set()

        self.sparsity = safe_divide(sdr_size, self.sds.size)
        self.relative_sparsity = safe_divide(sdr_size, self.sds.active_size)

        self.diff_relative_rate = safe_divide(len(sdr - prev_sdr), self.sds.active_size)
        self.sym_diff_rate = safe_divide(len(sdr ^ prev_sdr), len(sdr | prev_sdr))

        # aggregate/cluster/sequence metrics
        aggregate_union_size = np.count_nonzero(self.aggregate_histogram)
        aggregate_pmf = self.aggregate_pmf()

        self.aggregate_sparsity = safe_divide(aggregate_union_size, self.sds.size)
        self.aggregate_relative_sparsity = safe_divide(aggregate_union_size, self.sds.active_size)
        self.aggregate_entropy = entropy(aggregate_pmf, self.sds)

        # step coverage metrics
        covered_pmf = aggregate_pmf[list_sdr]
        covered_entropy = entropy(covered_pmf, self.sds)
        self.union_coverage = safe_divide(sdr_size, aggregate_union_size)
        self.pmf_coverage = safe_divide(covered_pmf.sum(), aggregate_pmf.sum())
        self.entropy_coverage = safe_divide(covered_entropy, self.aggregate_entropy)

        self.pmf_coverage_history.append(self.pmf_coverage)

    def step_metrics(self) -> TMetrics:
        return {
            'step/sparsity': self.sparsity,
            'step/relative_sparsity': self.relative_sparsity,
            'step/new_cells_relative_ratio': self.diff_relative_rate,
            'step/sym_diff_cells_ratio': self.sym_diff_rate,
            'coverage/union': self.union_coverage,
            'coverage/pmf': self.pmf_coverage,
            'coverage/entropy': self.entropy_coverage,
            'agg/sparsity': self.aggregate_sparsity,
            'agg/relative_sparsity': self.aggregate_relative_sparsity,
            'agg/entropy': self.aggregate_entropy,
        }

    def aggregate_metrics(self) -> TMetrics:
        return {
            'epoch/mean_pmf_coverage': np.mean(self.pmf_coverage_history)
        }
