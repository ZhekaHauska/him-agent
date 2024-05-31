#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from hima.experiments.temporal_pooling.stp.sp import SpNewbornPruningMode
from hima.experiments.temporal_pooling.stp.sp_utils import (
    RepeatingCountdown,
    make_repeating_counter
)


class PruningController:
    sp: Any

    mode: SpNewbornPruningMode
    schedule: int
    n_stages: int
    stage: int
    countdown: RepeatingCountdown

    def __init__(
            self, sp,
            mode: str, cycle: float, n_stages: int,
            target_rf_sparsity: float = None, target_rf_to_input_ratio: float = None
    ):
        self.sp = sp

        self.mode = SpNewbornPruningMode[mode.upper()]
        self.schedule = int(cycle / sp.output_sds.sparsity)
        self.n_stages = n_stages
        self.stage = 0
        self.countdown = make_repeating_counter(self.schedule)

        self.initial_rf_sparsity = 1.0
        self.target_rf_to_input_ratio = target_rf_to_input_ratio
        self.target_rf_sparsity = target_rf_sparsity

    @property
    def is_newborn_phase(self):
        return self.stage < self.n_stages

    def shrink_receptive_field(self):
        self.stage += 1

        if self.mode == SpNewbornPruningMode.LINEAR:
            new_sparsity = self.newborn_linear_progress(
                initial=self.initial_rf_sparsity, target=self.get_target_rf_sparsity()
            )
            print(self.initial_rf_sparsity, self.get_target_rf_sparsity(), new_sparsity)
        elif self.mode == SpNewbornPruningMode.POWERLAW:
            new_sparsity = self.newborn_powerlaw_progress(
                initial=self.sp.rf_sparsity, target=self.get_target_rf_sparsity()
            )
        else:
            raise ValueError(f'Pruning mode {self.mode} is not supported')

        if new_sparsity > self.sp.rf_sparsity:
            # if feedforward sparsity is tracked, then it may change and lead to RF increase
            # ==> leave RF as is
            return

        # sample what connections to keep for each neuron independently
        new_rf_size = round(new_sparsity * self.sp.ff_size)
        prune(self.sp.rng, self.sp.weights, new_rf_size)
        return new_sparsity, new_rf_size

    def get_target_rf_sparsity(self):
        if self.target_rf_sparsity is not None:
            return self.target_rf_sparsity

        if self.sp.adapt_to_ff_sparsity:
            ff_sparsity = self.sp.ff_avg_sparsity
        else:
            ff_sparsity = self.sp.feedforward_sds.sparsity

        return self.target_rf_to_input_ratio * ff_sparsity

    def newborn_linear_progress(self, initial, target):
        newborn_phase_progress = self.stage / self.n_stages
        # linear decay rule
        return initial + newborn_phase_progress * (target - initial)

    def newborn_powerlaw_progress(self, initial, target):
        steps_left = self.n_stages - self.stage + 1
        current = self.sp.rf_sparsity
        # what decay is needed to reach the target in the remaining steps
        # NB: recalculate each step to exclude rounding errors
        decay = np.power(target / current, 1 / steps_left)
        # exponential decay rule
        return current * decay


@numba.jit(nopython=True, cache=True)
def prune(rng: Generator, weights: npt.NDArray[float], k: int):
    n_neurons, n_synapses = weights.shape
    cache_mask = np.zeros(n_synapses, dtype=np.bool_)

    for row in range(n_neurons):
        abs_ws = np.abs(weights[row])
        nz_ws = np.flatnonzero(abs_ws)

        threshold = abs_ws[nz_ws].mean() + 1e-20
        keep_prob = (abs_ws[nz_ws] / threshold + 0.1) ** 1.4

        nb_choice_k(
            rng, k, keep_prob, None, False, cache_mask
        )
        pruned_ixs = nz_ws[~cache_mask[:nz_ws.size]]
        weights[row, pruned_ixs] = 0.
        cache_mask.fill(False)


@numba.jit(nopython=True, cache=True)
def nb_choice_k(
        rng: Generator, k: int, weights: npt.NDArray[np.float64] = None, n: int = None,
        replace: bool = False, cache: npt.NDArray[np.bool_] = None
):
    """Choose k samples from max_n values, with optional weights and replacement."""
    acc_w = np.cumsum(weights) if weights is not None else np.arange(0, n, 1, dtype=np.float64)
    # Total of weights
    mx_w = acc_w[-1]
    # result
    result = np.full(k, -1, dtype=np.int64)
    if not replace and cache is None and n is not None:
        cache = np.zeros(n, dtype=np.bool_)

    i = 0
    while i < k:
        r = mx_w * rng.random()
        ind = np.searchsorted(acc_w, r, side='right')

        if not replace and cache[ind]:
            continue
        else:
            result[i] = ind
            if not replace:
                cache[ind] = True
            i += 1

    return result
