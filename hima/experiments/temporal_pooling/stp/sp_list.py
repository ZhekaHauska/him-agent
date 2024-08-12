#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.sp_utils import sample_rf
from hima.experiments.temporal_pooling.stp.se_utils import boosting


class NewbornNeuron:
    """Neuron is a single neuron in the network."""
    id: int
    ff_sds: Sds
    potential_rf: np.ndarray
    _potential_rf: set
    rf: set
    weights: np.ndarray
    threshold = 0.3
    rng: Generator

    n_matches: float
    n_activations: float
    activation_heatmap: np.ndarray

    def __init__(
            self, id: int, ff_sds: Sds, initial_rf_sparsity: float, avg_rate: float,
            boosting_k: float, rng: Generator
    ):
        self.id = id
        self.ff_sds = ff_sds
        self.rng = rng
        self.potential_rf = sample_rf(ff_sds, initial_rf_sparsity, rng)
        self.weights = rng.uniform(0, 1, size=len(self.potential_rf))
        self.rf = set(self.potential_rf[self.weights >= self.threshold])
        self.target_avg_rate = avg_rate

        self.n_matches = 1
        self.n_activations = 1
        self.boosting_log_1_k = np.log(1.0 + boosting_k)
        self.activation_heatmap = np.ones(len(self.potential_rf))

    def match(self, input_sdr: SparseSdr) -> float:
        """Activate the neuron."""
        overlap = len(self.rf & input_sdr) * self.boosting()
        self.n_matches += 1
        return overlap

    @property
    def rate(self):
        return self.n_activations / self.n_matches

    def boosting(self) -> float:
        return boosting(
            relative_rate=self.target_avg_rate / self.rate,
            log_k=self.boosting_log_1_k
        )

    def activate(self, input_sdr: SparseSdr):
        """Activate the neuron."""
        self.n_activations += 1
        # matched_presynaptic_neurons = np.isin(self.potential_rf, input_sdr)
        for i in range(self.potential_rf.shape[0]):
            if self.potential_rf[i] in input_sdr:
                self.activation_heatmap[i] += 1

    def learn(self, input_sdr: SparseSdr, learning_rate_inc: float, learning_rate_dec: float):
        """Learn the neuron."""
        # global inhibition
        self.weights -= learning_rate_dec

        # strengthen connections to the current input
        for i in range(self.potential_rf.shape[0]):
            if self.potential_rf[i] in input_sdr:
                self.weights[i] += learning_rate_inc + learning_rate_dec
                if self.weights[i] > 0.5 and self.potential_rf[i] not in self.rf:
                    self.rf.add(self.potential_rf[i])
                if self.weights[i] > 1:
                    self.weights[i] = 1

        # as decay is much slower and negligible, we may skip several updates
        if self.n_activations % 5 == 0:
            np.clip(self.weights, 0, 1, out=self.weights)
            self.rf = set(self.potential_rf[self.weights >= self.threshold])

    def prune_rf(self, new_rf_sparsity: float):
        """Prune the receptive field."""
        new_rf_size = int(self.ff_sds.size * new_rf_sparsity)
        keep_prob = np.power(self.activation_heatmap, 2.0)
        keep_prob /= keep_prob.sum()

        keep_connections = self.rng.choice(
            self.potential_rf.shape[0], size=new_rf_size, replace=False,
            p=keep_prob
        )
        self.potential_rf = self.potential_rf[keep_connections].copy()
        self.weights = self.weights[keep_connections].copy()
        self.activation_heatmap = self.activation_heatmap[keep_connections].copy()
        self.rf = set(self.potential_rf[self.weights >= self.threshold])
        # self.boosting_k /= 2


class SpatialPooler:
    """
    Spatial Pooler implementation with receptive fields stored in lists.
    Thus, this is a non-vectorized [over neurons] SP implementation.
    """

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
    neurons: list[NewbornNeuron]

    n_computes: int
    cum_input_size: int
    newborn_pruning_cycle: float
    newborn_pruning_stages: int
    _newborn_prune_iteration: int

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

        self._rng = np.random.default_rng(seed)
        self.neurons = [
            NewbornNeuron(
                id, self.feedforward_sds, initial_rf_sparsity, self.output_sds.sparsity,
                boosting_k, self._rng
            )
            for id in range(self.output_sds.size)
        ]

        self.newborn_pruning_cycle = newborn_pruning_cycle
        self.newborn_pruning_stages = newborn_pruning_stages
        self._newborn_prune_iteration = 0
        self.n_computes = 0
        self.cum_input_size = 0
        self.run_time = 0

    def compute(self, input_sdr: SparseSdr, learn: bool = False) -> SparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.run_time += run_time
        return output_sdr

    @timed
    def _compute(self, input_sdr: SparseSdr, learn: bool = True):
        """Compute the output SDR."""
        input_sdr_set = set(input_sdr)
        # start_time = time.time()
        overlaps = np.array([
            neuron.match(input_sdr_set)
            for neuron in self.neurons
        ])
        # run_time = time.time() - start_time
        n_winners = self.output_sds.active_size
        winners = np.sort(
            np.argpartition(-overlaps, n_winners)[:n_winners]
        )
        for winner in winners:
            self.neurons[winner].activate(input_sdr_set)
            if learn:
                self.neurons[winner].learn(
                    input_sdr_set, self.learning_rate_inc, self.learning_rate_dec
                )

        self.n_computes += 1
        self.cum_input_size += len(input_sdr_set)
        if self.n_computes % int(self.newborn_pruning_cycle * self.output_sds.size) == 0:
            self.prune_newborns()
        return winners #, run_time

    def prune_newborns(self):
        if self._newborn_prune_iteration == self.newborn_pruning_stages:
            self._newborn_prune_iteration += 1
            rf_size = int(self.rf_sparsity * self.feedforward_sds.size)
            print(f'Turning off newborns: {self.rf_sparsity} | {rf_size}')
            for neuron in self.neurons:
                neuron.boosting_log_1_k = 0.0
                self.learning_rate_inc /= 2
                self.learning_rate_dec /= 2
            return
        if self._newborn_prune_iteration > self.newborn_pruning_stages:
            return

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

        rf_size = int(self.rf_sparsity * self.feedforward_sds.size)
        print(f'Pruning newborns: {self.rf_sparsity} | {rf_size}')
        for neuron in self.neurons:
            neuron.prune_rf(self.rf_sparsity)

    def activation_entropy(self):
        activation_heatmap = np.array([
            neuron.rate for neuron in self.neurons
        ])
        activation_probs = activation_heatmap
        return (
            entropy(activation_probs, sds=self.output_sds),
            # np.round(activation_probs / self.output_sds.sparsity, 2)
        )
