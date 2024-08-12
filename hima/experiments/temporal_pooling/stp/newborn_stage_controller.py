#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np

from hima.common.utils import isnone
from hima.experiments.temporal_pooling.stp.sp import SpNewbornPruningMode
from hima.experiments.temporal_pooling.stp.sp_utils import (
    sample_for_each_neuron, gather_rows,
    normalize_weights
)
from hima.experiments.temporal_pooling.stp.se_utils import boosting


class NewbornStageController:
    sp: Any

    newborn_pruning_mode: SpNewbornPruningMode
    newborn_pruning_stages: int
    newborn_pruning_schedule: int
    newborn_pruning_stage: int

    base_boosting_k: float
    boosting_k: float

    def __init__(
            self, sp,
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            newborn_pruning_mode: str, boosting_k: float,

            initial_max_rf_sparsity: float, target_max_rf_sparsity: float,
            initial_rf_to_input_ratio: float, target_rf_to_input_ratio: float,
            connectable_ff_size: int = None,
    ):
        self.sp = sp

        self.newborn_pruning_mode = SpNewbornPruningMode[newborn_pruning_mode.upper()]
        self.newborn_pruning_schedule = int(newborn_pruning_cycle / sp.output_sds.sparsity)
        self.newborn_pruning_stages = newborn_pruning_stages
        self.newborn_pruning_stage = 0
        self.base_boosting_k = boosting_k
        self.boosting_k = self.base_boosting_k

        self.target_rf_to_input_ratio = target_rf_to_input_ratio
        self.target_max_rf_sparsity = target_max_rf_sparsity

        if self.is_newborn_phase:
            self.initial_rf_sparsity = min(
                initial_rf_to_input_ratio * sp.feedforward_sds.sparsity,
                initial_max_rf_sparsity
            )
        else:
            self.initial_rf_sparsity = self.get_target_rf_sparsity()

        rf_size = int(self.initial_rf_sparsity * sp.ff_size)
        if not isinstance(connectable_ff_size, int):
            connectable_ff_size = None
        set_size = isnone(connectable_ff_size, sp.ff_size)
        rf_size = min(rf_size, set_size)

        self.sp.rf = sample_for_each_neuron(
            rng=sp.rng, n_neurons=sp.output_size,
            set_size=set_size, sample_size=rf_size
        )
        print(f'SP.layer init shape: {self.sp.rf.shape} to {set_size}')

    @property
    def is_newborn_phase(self):
        return self.newborn_pruning_stage < self.newborn_pruning_stages

    def shrink_receptive_field(self):
        self.newborn_pruning_stage += 1

        if self.newborn_pruning_mode == SpNewbornPruningMode.LINEAR:
            new_sparsity = self.newborn_linear_progress(
                initial=self.initial_rf_sparsity, target=self.get_target_rf_sparsity()
            )
        elif self.newborn_pruning_mode == SpNewbornPruningMode.POWERLAW:
            new_sparsity = self.newborn_powerlaw_progress(
                initial=self.sp.rf_sparsity, target=self.get_target_rf_sparsity()
            )
        else:
            raise ValueError(f'Pruning mode {self.newborn_pruning_mode} is not supported')

        if new_sparsity > self.sp.rf_sparsity:
            # if feedforward sparsity is tracked, then it may change and lead to RF increase
            # ==> leave RF as is
            return

        # probabilities to keep connection
        threshold = .5 / self.sp.rf_size
        keep_prob = np.power(np.abs(self.sp.weights) / threshold + 0.1, 2.0)
        keep_prob /= keep_prob.sum(axis=1, keepdims=True)

        # sample what connections to keep for each neuron independently
        new_rf_size = round(new_sparsity * self.sp.ff_size)
        keep_connections_i = sample_for_each_neuron(
            rng=self.sp.rng, n_neurons=self.sp.output_size,
            set_size=self.sp.rf_size, sample_size=new_rf_size, probs_2d=keep_prob
        )

        self.sp.rf = gather_rows(self.sp.rf, keep_connections_i)
        self.sp.weights = normalize_weights(
            gather_rows(self.sp.weights, keep_connections_i)
        )
        self.boosting_k = self.newborn_linear_progress(
            initial=self.base_boosting_k, target=0.
        )
        if not self.is_newborn_phase:
            self.boosting_k = 0

        print(f'Prune newborns: {self.sp.sng_state_str()}')

    def apply_boosting(self, overlaps):
        if self.is_newborn_phase and self.boosting_k > 1e-2:
            # boosting
            boosting_alpha = boosting(relative_rate=self.sp.output_relative_rate, k=self.boosting_k)
            # FIXME: normalize boosting alpha over neurons
            overlaps *= boosting_alpha

    def get_target_rf_sparsity(self):
        if self.sp.adapt_to_ff_sparsity:
            ff_sparsity = self.sp.ff_avg_sparsity
        else:
            ff_sparsity = self.sp.feedforward_sds.sparsity

        return min(
            self.target_rf_to_input_ratio * ff_sparsity,
            self.target_max_rf_sparsity,
        )

    def newborn_linear_progress(self, initial, target):
        newborn_phase_progress = self.newborn_pruning_stage / self.newborn_pruning_stages
        # linear decay rule
        return initial + newborn_phase_progress * (target - initial)

    def newborn_powerlaw_progress(self, initial, target):
        steps_left = self.newborn_pruning_stages - self.newborn_pruning_stage + 1
        current = self.sp.rf_sparsity
        # what decay is needed to reach the target in the remaining steps
        # NB: recalculate each step to exclude rounding errors
        decay = np.power(target / current, 1 / steps_left)
        # exponential decay rule
        return current * decay
