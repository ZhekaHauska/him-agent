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


class NewbornNeuron:
    """Neuron is a single neuron in the network."""
    id: int
    sds: Sds
    potential_rf: np.ndarray
    rf: SparseSdr
    weights: np.ndarray
    threshold = 0.3
    rng: Generator

    n_matches: float
    n_activations: float
    activation_heatmap: np.ndarray

    def __init__(
            self, id: int, sds: Sds, initial_rf_sparsity: float, avg_rate: float,
            rng: Generator
    ):
        self.id = id
        self.sds = sds
        self.rng = rng
        self.potential_rf = sample_rf(sds, initial_rf_sparsity, rng)
        self.weights = rng.uniform(0, 1, size=len(self.potential_rf))
        self.rf = set(self.potential_rf[self.weights >= self.threshold])
        self.target_avg_rate = avg_rate

        self.n_matches = 1
        self.n_activations = 1
        self.boosting_k = 2.0
        self.activation_heatmap = np.ones(len(self.potential_rf))

    def match(self, input_sdr: SparseSdr, avg_rate: float) -> float:
        """Activate the neuron."""
        overlap = len(self.rf & input_sdr) * self.boosting()
        self.n_matches += 1
        # if self.rate / self.target_avg_rate < 0.
        return overlap

    @property
    def rate(self):
        return self.n_activations / self.n_matches

    def boosting(self) -> float:
        boosting = self.rate / self.target_avg_rate
        # print(self.rate > self.target_avg_rate, boosting, np.exp(-k * boosting))
        return np.exp(-self.boosting_k * boosting)

    def activate(self, input_sdr: SparseSdr):
        """Activate the neuron."""
        self.n_activations += 1
        matched_presynaptic_neurons = np.isin(self.potential_rf, input_sdr)
        self.activation_heatmap[matched_presynaptic_neurons] += 1

    def learn(self, input_sdr: SparseSdr, learning_rate_inc: float, learning_rate_dec: float):
        """Learn the neuron."""
        matched_presynaptic_neurons = np.isin(self.potential_rf, input_sdr)
        self.weights[matched_presynaptic_neurons] += learning_rate_inc
        self.weights[~matched_presynaptic_neurons] -= learning_rate_dec
        np.clip(self.weights, 0, 1, out=self.weights)

        self.rf = set(self.potential_rf[self.weights >= self.threshold])

    def prune_rf(self, new_rf_sparsity: float):
        """Prune the receptive field."""
        new_rf_size = int(self.sds.size * new_rf_sparsity)
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


def sample_rf(sds: Sds, rf_sparsity: float, rng: Generator):
    """Sample a random receptive field."""
    rf_size = int(sds.size * rf_sparsity)
    return rng.choice(sds.size, rf_size, replace=False)


class SpatialPooler:
    # input
    ff_sds: Sds
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
            self, ff_sds: Sds,
            initial_rf_sparsity: float, max_rf_sparsity: float, max_rf_to_input_ratio: float,
            output_sds: Sds,
            min_overlap_for_activation: float, learning_rate_inc: float, learning_rate_dec: float,
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            seed: int
    ):
        self.ff_sds = ff_sds
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
                id, self.ff_sds, initial_rf_sparsity, self.output_sds.sparsity, self._rng
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
        overlaps = np.array([
            neuron.match(input_sdr_set, self.output_sds.sparsity) for neuron in self.neurons
        ])
        # print(overlaps)
        n_winners = self.output_sds.active_size
        winners = np.sort(
            np.argpartition(-overlaps, n_winners)[:n_winners]
        )
        # print(winners)

        for winner in winners:
            self.neurons[winner].activate(input_sdr)

        if learn:
            for winner in winners:
                self.neurons[winner].learn(
                    input_sdr, self.learning_rate_inc, self.learning_rate_dec
                )

        self.n_computes += 1
        self.cum_input_size += len(input_sdr_set)
        if self.n_computes % int(self.newborn_pruning_cycle * self.output_sds.size) == 0:
            self.prune_newborns()
        return winners

    def prune_newborns(self):
        if self._newborn_prune_iteration == self.newborn_pruning_stages:
            self._newborn_prune_iteration += 1
            print('Turning off newborns')
            for neuron in self.neurons:
                neuron.boosting_k = 0.0
                self.learning_rate_inc /= 2
                self.learning_rate_dec /= 2
            return
        if self._newborn_prune_iteration > self.newborn_pruning_stages:
            return

        print('Pruning newborns')
        avg_input_size = self.cum_input_size / self.n_computes
        input_sparsity = avg_input_size / self.ff_sds.size

        target_rf_sparsity = min(
            self._max_rf_sparsity,
            self._max_rf_to_input_ratio * input_sparsity
        )
        self._newborn_prune_iteration += 1
        new_rf_sparsity = self._initial_rf_sparsity + self._newborn_prune_iteration * (
            target_rf_sparsity - self._initial_rf_sparsity
        ) / self.newborn_pruning_stages
        for neuron in self.neurons:
            neuron.prune_rf(new_rf_sparsity)

    def activation_entropy(self):
        activation_heatmap = np.array([
            neuron.rate for neuron in self.neurons
        ])
        activation_probs = activation_heatmap
        return (
            entropy(activation_probs, sds=self.output_sds),
            # np.round(activation_probs / self.output_sds.sparsity, 2)
        )
