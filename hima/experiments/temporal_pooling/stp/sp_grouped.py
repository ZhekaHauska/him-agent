#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import timed, safe_divide
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.sp import SpatialPooler
from hima.experiments.temporal_pooling.stp.sp_utils import (
    boosting, gather_rows,
    sample_for_each_neuron
)


class SpatialPoolerGrouped(SpatialPooler):
    rng: Generator

    # input
    feedforward_sds: Sds

    initial_rf_sparsity: float
    max_rf_sparsity: float
    max_rf_to_input_ratio: float

    # output
    output_sds: Sds
    min_overlap_for_activation: float

    # learning
    learning_rate: float

    # connections
    newborn_pruning_cycle: float
    newborn_pruning_stages: int
    newborn_pruning_mode: str
    newborn_pruning_stage: int
    prune_grow_cycle: float

    # stats
    n_computes: int
    feedforward_trace: np.ndarray
    output_trace: np.ndarray

    # vectorized fields
    rf: np.ndarray
    weights: np.ndarray
    threshold = 0.3
    base_boosting_k: float
    output_trace: np.ndarray

    def __init__(
            self, *,
            feedforward_sds: Sds,
            # newborn / mature
            initial_rf_to_input_ratio: float, max_rf_to_input_ratio: float, max_rf_sparsity: float,
            output_sds: Sds,
            min_overlap_for_activation: float, learning_rate: float,
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            prune_grow_cycle: float,
            boosting_k: float, seed: int,
            adapt_to_ff_sparsity: bool = True,
            newborn_pruning_mode: str = 'powerlaw'
    ):
        super().__init__(
            feedforward_sds=feedforward_sds, initial_rf_to_input_ratio=initial_rf_to_input_ratio,
            max_rf_to_input_ratio=max_rf_to_input_ratio, max_rf_sparsity=max_rf_sparsity,
            output_sds=output_sds, min_overlap_for_activation=min_overlap_for_activation,
            learning_rate=learning_rate, newborn_pruning_cycle=newborn_pruning_cycle,
            newborn_pruning_stages=newborn_pruning_stages, prune_grow_cycle=prune_grow_cycle,
            boosting_k=boosting_k, seed=seed, adapt_to_ff_sparsity=adapt_to_ff_sparsity,
            newborn_pruning_mode=newborn_pruning_mode
        )
        assert self.output_size % self.n_groups == 0, f'Non-divisible groups {self.output_sds}'
        self.group_size = self.output_sds.size // self.n_groups
        self.group_shifts = np.arange(self.n_groups) * self.group_size

    def compute_winners(self, overlaps):
        overlaps_grouped = overlaps.reshape(self.n_groups, -1)
        winners_grouped = np.argpartition(overlaps_grouped, -1, axis=-1)[:, -1:]
        return winners_grouped.flatten() + self.group_shifts

    @property
    def n_groups(self):
        return self.output_sds.active_size
