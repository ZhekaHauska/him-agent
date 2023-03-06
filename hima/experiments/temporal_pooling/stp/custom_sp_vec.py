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
from hima.experiments.temporal_pooling.stp.custom_sp_utils import sample_rf, boosting


class SpatialPooler:
    # input
    feedforward_sds: Sds
    rf_sparsity: float

    _initial_rf_sparsity: float
    _max_rf_sparsity: float
    _max_rf_to_input_ratio: float

    # output
    output_sds: Sds

    # learning
    min_overlap_for_activation: float
    learning_rate_inc: float
    learning_rate_dec: float

    _rng: Generator

    # connections
    n_computes: int
    cum_input_size: int
    newborn_pruning_cycle: float
    newborn_pruning_stages: int
    _newborn_prune_iteration: int

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
            min_overlap_for_activation: float, learning_rate_inc: float, learning_rate_dec: float,
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            boosting_k: float,
            seed: int
    ):
        self.feedforward_sds = feedforward_sds
        self.rf_sparsity = initial_rf_sparsity
        self._initial_rf_sparsity = initial_rf_sparsity
        self._max_rf_sparsity = max_rf_sparsity
        self._max_rf_to_input_ratio = max_rf_to_input_ratio

        self.output_sds = output_sds

        self.min_overlap_for_activation = min_overlap_for_activation
        self.learning_rate_inc = learning_rate_inc
        self.learning_rate_dec = learning_rate_dec
        self.polarity = 1

        self._rng = np.random.default_rng(seed)

        self.potential_rf = np.array([
            sample_rf(feedforward_sds, self.rf_sparsity, self._rng)
            for _ in range(self.output_sds.size)
        ])
        print(f'SP vec init shape: {self.potential_rf.shape}')

        self.weights = self._rng.uniform(0, 1, size=self.potential_rf.shape)
        self.rf = self.weights >= self.threshold
        self.sparse_input = []
        self.dense_input = np.zeros(self.feedforward_sds.size, dtype=int)

        self.n_activations = np.ones(self.output_sds.size)
        self.boosting_log_1_k = np.log(1.0 + boosting_k)
        self.activation_heatmap = np.ones(self.potential_rf.shape)

        self.newborn_pruning_cycle = newborn_pruning_cycle
        self.newborn_pruning_stages = newborn_pruning_stages
        self._newborn_prune_iteration = 0
        self.n_computes = 0
        self.cum_input_size = 0
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
        self.cum_input_size += len(input_sdr)

        self.dense_input[self.sparse_input] = 0
        self.sparse_input = input_sdr
        self.dense_input[self.sparse_input] = 1

        matches = self.dense_input[self.potential_rf]
        matches_active = matches & self.rf
        overlaps = matches_active.sum(axis=1)

        # boosting
        rates = self.n_activations / self.n_computes
        target_rate = self.output_sds.sparsity
        boosting_alpha = boosting(relative_rate=target_rate / rates, log_k=self.boosting_log_1_k)
        overlaps = overlaps * boosting_alpha

        n_winners = self.output_sds.active_size
        winners = np.sort(
            np.argpartition(-overlaps, n_winners)[:n_winners]
        )

        # update winners activation stats
        self.n_activations[winners] += 1
        self.activation_heatmap[winners] += matches[winners]

        if learn:
            self.learn(winners, matches_active[winners])

        if self.n_computes % int(self.newborn_pruning_cycle * self.output_sds.size) == 0:
            self.prune_newborns()
        return winners

    def learn(self, winners, winners_active_matches):
        # global inhibition + strengthen connections to the current input
        lr_dec = self.learning_rate_dec
        lr_inc = self.learning_rate_inc + self.learning_rate_dec
        if self.polarity < 0:
            lr_inc, lr_dec = -lr_inc, -lr_dec

        self.weights[winners] = np.clip(
            self.weights[winners] - lr_dec + lr_inc * winners_active_matches,
            0, 1
        )
        self.rf[winners] = self.weights[winners] >= self.threshold

    def process_feedback(self, feedback_sdr: SparseSdr):
        # TODO: rename to feedback
        feedback_matches = self.dense_input[
            self.potential_rf[feedback_sdr]
        ]
        feedback_active_matches = feedback_matches & self.rf[feedback_sdr]
        k = 5
        self.polarity *= k
        self.learn(feedback_sdr, feedback_active_matches)
        self.polarity /= k

    def prune_newborns(self):
        if self._newborn_prune_iteration == self.newborn_pruning_stages:
            self._newborn_prune_iteration += 1
            self.boosting_log_1_k = 0.
            self.learning_rate_inc /= 2
            self.learning_rate_dec /= 2
            print(f'Boosting off: {self._state_str}')
            return

        if self._newborn_prune_iteration > self.newborn_pruning_stages:
            return

        self.update_rf_size()
        print(f'Prune newborns: {self._state_str}')
        self.prune_rf()

    def update_rf_size(self):
        """Update the receptive field size."""
        avg_input_size = self.cum_input_size / self.n_computes
        input_sparsity = avg_input_size / self.feedforward_sds.size

        target_rf_sparsity = min(
            self._max_rf_sparsity,
            self._max_rf_to_input_ratio * input_sparsity
        )
        self._newborn_prune_iteration += 1
        self.rf_sparsity = self._initial_rf_sparsity + self._newborn_prune_iteration * (
            target_rf_sparsity - self._initial_rf_sparsity
        ) / self.newborn_pruning_stages

    def prune_rf(self):
        """Prune the receptive field."""
        keep_prob = np.power(self.activation_heatmap, 2.0)
        keep_prob /= keep_prob.sum(axis=1, keepdims=True)

        # sample what connections to keep for each neuron independently
        keep_connections_i = np.array([
            self._rng.choice(
                self.potential_rf.shape[1], size=self.rf_size, replace=False,
                p=keep_prob[neuron]
            )
            for neuron in range(self.output_sds.size)
        ])

        def gather_rows(arr_2d, indices):
            return np.array([
                arr_2d[neuron, indices[neuron]]
                for neuron in range(self.output_sds.size)
            ])

        self.potential_rf = gather_rows(self.potential_rf, keep_connections_i)
        self.weights = gather_rows(self.weights, keep_connections_i)
        self.activation_heatmap = gather_rows(self.activation_heatmap, keep_connections_i)
        self.rf = self.weights >= self.threshold

    def activation_entropy(self):
        activation_probs = self.n_activations / self.n_computes
        return (
            entropy(activation_probs, sds=self.output_sds),
            # np.round(activation_probs / self.output_sds.sparsity, 2)
        )

    @property
    def rf_size(self):
        return int(self.rf_sparsity * self.feedforward_sds.size)

    @property
    def _state_str(self) -> str:
        return f'{self.rf_sparsity:.4f} | {self.rf_size} | {self.learning_rate_inc:.4f}'
