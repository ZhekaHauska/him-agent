#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import timed
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.sp_utils import (
    boosting, gather_rows,
    sample_for_each_neuron
)


class SpatialPooler:
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
    global_inhibition_strength: float

    # connections
    newborn_pruning_cycle: float
    newborn_pruning_stages: int
    newborn_pruning_stage: int

    # stats
    n_computes: int
    cumulative_active_input_size: int

    # vectorized fields
    potential_rf: np.ndarray
    rf: np.ndarray
    weights: np.ndarray
    threshold = 0.3
    boosting_log_1_k: float
    n_activations: np.ndarray
    activation_heatmap: np.ndarray

    def __init__(
            self, feedforward_sds: Sds,
            initial_rf_sparsity: float, max_rf_sparsity: float, max_rf_to_input_ratio: float,
            output_sds: Sds,
            min_overlap_for_activation: float,
            learning_rate: float, global_inhibition_strength: float,
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            boosting_k: float, seed: int,
            adapt_to_ff_sparsity: bool = True,
    ):
        self.rng = np.random.default_rng(seed)
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds

        self.adapt_to_ff_sparsity = adapt_to_ff_sparsity

        self.initial_rf_sparsity = initial_rf_sparsity
        self.max_rf_sparsity = max_rf_sparsity
        self.max_rf_to_input_ratio = max_rf_to_input_ratio

        self.min_overlap_for_activation = min_overlap_for_activation
        self.learning_rate = learning_rate
        self.global_inhibition_strength = global_inhibition_strength
        self.polarity = 1

        rf_size = int(self.initial_rf_sparsity * self.ff_size)
        self.potential_rf = sample_for_each_neuron(
            rng=self.rng, n_neurons=self.output_sds.size,
            set_size=self.ff_size, sample_size=rf_size
        )
        print(f'SP vec init shape: {self.potential_rf.shape}')

        self.weights = self.rng.uniform(0, 1, size=self.potential_rf.shape)
        self.rf = self.weights >= self.threshold

        self.sparse_input = []
        self.dense_input = np.zeros(self.ff_size, dtype=int)

        self.n_activations = np.ones(self.output_sds.size)
        self.boosting_log_1_k = np.log(1.0 + boosting_k)
        self.activation_heatmap = np.ones(self.potential_rf.shape)

        self.newborn_pruning_cycle = newborn_pruning_cycle
        self.newborn_pruning_stages = newborn_pruning_stages
        self.newborn_pruning_stage = 0
        self.n_computes = 0
        self.no_feedback_count = 0
        self.cumulative_active_input_size = 0
        self.run_time = 0

    def compute(self, input_sdr: SparseSdr, learn: bool = False) -> SparseSdr:
        """Compute the output SDR."""
        # TODO: rename to feedforward
        output_sdr, run_time = self._compute_for_newborn(input_sdr, learn)
        self.run_time += run_time
        return output_sdr

    @timed
    def _compute_for_newborn(self, input_sdr: SparseSdr, learn: bool) -> SparseSdr:
        self.n_computes += 1
        self.no_feedback_count += 1
        if self.n_computes % int(self.newborn_pruning_cycle * self.output_sds.size) == 0:
            self.shrink_receptive_field()

        self.cumulative_active_input_size += len(input_sdr)
        self.update_input(input_sdr)

        match_mask, active_match_mask = self.match_input(self.dense_input)
        overlaps = active_match_mask.sum(axis=1)

        if self.is_boosting_on:
            # boosting
            rates = self.n_activations / self.n_computes
            target_rate = self.output_sds.sparsity
            relative_rates = target_rate / rates
            boosting_alpha = boosting(relative_rate=relative_rates, log_k=self.boosting_log_1_k)
            overlaps = overlaps * boosting_alpha

        n_winners = self.output_sds.active_size
        winners = np.sort(
            np.argpartition(-overlaps, n_winners)[:n_winners]
        )

        # update winners activation stats
        self.n_activations[winners] += 1
        self.activation_heatmap[winners] += match_mask[winners]

        if learn:
            self.learn(winners, match_mask[winners])
        return winners

    def learn(
            self, neurons: np.ndarray, match_input_mask: np.ndarray,
            strength: float = 1.0
    ):
        all_ = self.global_inhibition_strength
        matched = 1. + all_

        lr = strength * self.polarity * self.learning_rate

        self.weights[neurons] = np.clip(
            self.weights[neurons] + lr * (matched * match_input_mask - all_),
            0, 1
        )
        self.rf[neurons] = self.weights[neurons] >= self.threshold

    def process_feedback(self, feedback_sdr: SparseSdr):
        # feedback SDR is the SP neurons that should be reinforced
        feedback_strength = self.no_feedback_count
        fb_match_mask, _ = self.match_input(self.dense_input, neurons=feedback_sdr)

        self.learn(feedback_sdr, fb_match_mask, strength=feedback_strength)
        self.no_feedback_count = 0

    def shrink_receptive_field(self):
        if self.newborn_pruning_stage >= self.newborn_pruning_stages:
            return
        self.newborn_pruning_stage += 1

        new_sparsity = self.current_rf_sparsity()
        if new_sparsity > self.rf_sparsity:
            # if feedforward sparsity is tracked, then it may change and lead to RF increase
            # ==> leave RF as is
            return

        # probabilities to keep connection
        keep_prob = np.power(self.activation_heatmap, 2.0)
        keep_prob /= keep_prob.sum(axis=1, keepdims=True)

        # sample what connections to keep for each neuron independently
        new_rf_size = int(new_sparsity * self.ff_size)
        keep_connections_i = sample_for_each_neuron(
            rng=self.rng, n_neurons=self.output_sds.size,
            set_size=self.rf_size, sample_size=new_rf_size, probs_2d=keep_prob
        )

        self.potential_rf = gather_rows(self.potential_rf, keep_connections_i)
        self.weights = gather_rows(self.weights, keep_connections_i)
        self.activation_heatmap = gather_rows(self.activation_heatmap, keep_connections_i)
        self.rf = self.weights >= self.threshold
        print(f'Prune newborns: {self._state_str()}')

        if self.newborn_pruning_stage == self.newborn_pruning_stages:
            self.on_end_pruning_newborns()

    def on_end_pruning_newborns(self):
        self.boosting_log_1_k = 0.
        self.learning_rate /= 2
        self.global_inhibition_strength /= 2
        print(f'Boosting off: {self._state_str()}')

    def update_input(self, sdr: SparseSdr):
        # erase prev SDR
        self.dense_input[self.sparse_input] = 0
        # set new SDR
        self.sparse_input = sdr
        self.dense_input[self.sparse_input] = 1

    def match_input(self, dense_input, neurons: np.ndarray = None):
        if neurons is None:
            potential_rf, rf = self.potential_rf, self.rf
        else:
            potential_rf, rf = self.potential_rf[neurons], self.rf[neurons]

        match_mask = dense_input[potential_rf]
        active_match_mask = match_mask & rf
        return match_mask, active_match_mask

    def current_rf_sparsity(self):
        ff_sparsity = (
            self.ff_avg_sparsity if self.adapt_to_ff_sparsity else self.feedforward_sds.sparsity
        )
        final_rf_sparsity = min(
            self.max_rf_sparsity,
            self.max_rf_to_input_ratio * ff_sparsity
        )

        progress = self.newborn_pruning_stage / self.newborn_pruning_stages
        initial, final = self.initial_rf_sparsity, final_rf_sparsity
        return initial + progress * (final - initial)

    def activation_entropy(self):
        activation_probs = self.n_activations / self.n_computes
        return (
            entropy(activation_probs, sds=self.output_sds),
            # np.round(activation_probs / self.output_sds.sparsity, 2)
        )

    @property
    def ff_size(self):
        return self.feedforward_sds.size

    @property
    def ff_avg_active_size(self):
        return self.cumulative_active_input_size // self.n_computes

    @property
    def ff_avg_sparsity(self):
        return self.ff_avg_active_size / self.ff_size

    @property
    def rf_size(self):
        return self.rf.shape[1]

    @property
    def rf_sparsity(self):
        return self.rf_size / self.ff_size

    @property
    def is_boosting_on(self):
        return np.isclose(self.boosting_log_1_k, 0)

    def _state_str(self) -> str:
        return f'{self.rf_sparsity:.4f} | {self.rf_size} | {self.learning_rate:.4f}'
