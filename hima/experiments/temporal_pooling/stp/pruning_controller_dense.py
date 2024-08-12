#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import jit
from numpy.random import Generator

from hima.experiments.temporal_pooling.stp.sp import SpNewbornPruningMode
from hima.experiments.temporal_pooling.stp.sp_utils import (
    RepeatingCountdown,
    make_repeating_counter, nb_choice_k
)


class PruningController:
    owner: Any

    mode: SpNewbornPruningMode
    schedule: int
    n_stages: int
    stage: int
    countdown: RepeatingCountdown

    def __init__(
            self, owner,
            mode: str, cycle: float, n_stages: int,
            target_rf_sparsity: float = None, target_rf_to_input_ratio: float = None
    ):
        self.owner = owner

        self.mode = SpNewbornPruningMode[mode.upper()]
        self.schedule = int(cycle / owner.output_sds.sparsity)
        self.n_stages = n_stages
        self.stage = 0
        self.countdown = make_repeating_counter(self.schedule)

        self.initial_rf_sparsity = 1.0
        self.target_rf_to_input_ratio = target_rf_to_input_ratio
        self.target_rf_sparsity = target_rf_sparsity

    @property
    def is_newborn_phase(self):
        return self.stage < self.n_stages

    def shrink_receptive_field(self, pruned_mask):
        self.stage += 1

        if self.mode == SpNewbornPruningMode.LINEAR:
            new_sparsity = self.newborn_linear_progress(
                initial=self.initial_rf_sparsity, target=self.get_target_rf_sparsity()
            )
            print(self.initial_rf_sparsity, self.get_target_rf_sparsity(), new_sparsity)
        elif self.mode == SpNewbornPruningMode.POWERLAW:
            new_sparsity = self.newborn_powerlaw_progress(
                initial=self.owner.rf_sparsity, target=self.get_target_rf_sparsity()
            )
        else:
            raise ValueError(f'Pruning mode {self.mode} is not supported')

        if new_sparsity > self.owner.rf_sparsity:
            # if feedforward sparsity is tracked, then it may change and lead to RF increase
            # ==> leave RF as is
            return

        # sample what connections to keep for each neuron independently
        new_rf_size = round(new_sparsity * self.owner.ff_size)
        prune(self.owner.rng, self.owner.weights, new_rf_size, pruned_mask)
        return new_sparsity, new_rf_size

    def get_target_rf_sparsity(self):
        if self.target_rf_sparsity is not None:
            return self.target_rf_sparsity

        if self.owner.adapt_to_ff_sparsity:
            ff_sparsity = self.owner.ff_avg_sparsity
        else:
            ff_sparsity = self.owner.feedforward_sds.sparsity

        return self.target_rf_to_input_ratio * ff_sparsity

    def newborn_linear_progress(self, initial, target):
        newborn_phase_progress = self.stage / self.n_stages
        # linear decay rule
        return initial + newborn_phase_progress * (target - initial)

    def newborn_powerlaw_progress(self, initial, target):
        steps_left = self.n_stages - self.stage + 1
        current = self.owner.rf_sparsity
        # what decay is needed to reach the target in the remaining steps
        # NB: recalculate each step to exclude rounding errors
        decay = np.power(target / current, 1 / steps_left)
        # exponential decay rule
        return current * decay


@jit()
def prune(rng: Generator, weights: npt.NDArray[float], k: int, pruned_mask):
    n_neurons, n_synapses = weights.shape

    for row in range(n_neurons):
        pm_row = pruned_mask[row]
        w_row = weights[row]

        active_mask = ~pm_row
        abs_ws = np.abs(w_row[active_mask]) + 1e-20
        t = abs_ws.mean()
        prune_probs = (t / abs_ws + 0.1) ** 1.4

        # pruned connections are marked as already selected for "select K from N" operation
        n_active = len(prune_probs)
        not_k = n_active - k
        ixs = nb_choice_k(rng, not_k, prune_probs, n_active, False)
        new_pruned_ixs = np.flatnonzero(active_mask)[ixs]
        w_row[new_pruned_ixs] = 0.0
        pm_row[new_pruned_ixs] = True
