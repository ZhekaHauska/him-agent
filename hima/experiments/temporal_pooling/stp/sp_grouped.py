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


class SpatialPoolerGrouped:
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
            self, feedforward_sds: Sds,
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
        self.rng = np.random.default_rng(seed)
        self.feedforward_sds = Sds.make(feedforward_sds)
        self.output_sds = Sds.make(output_sds)

        self.adapt_to_ff_sparsity = adapt_to_ff_sparsity

        self.initial_rf_sparsity = min(
            initial_rf_to_input_ratio * self.feedforward_sds.sparsity,
            0.75
        )
        self.max_rf_to_input_ratio = max_rf_to_input_ratio
        self.max_rf_sparsity = max_rf_sparsity

        self.min_overlap_for_activation = min_overlap_for_activation
        self.learning_rate = learning_rate

        rf_size = int(self.initial_rf_sparsity * self.ff_size)
        self.rf = sample_for_each_neuron(
            rng=self.rng, n_neurons=self.output_size,
            set_size=self.ff_size, sample_size=rf_size
        )
        print(f'SP vec init shape: {self.rf.shape}')

        self.weights = self.normalize_weights(
            self.rng.normal(loc=1.0, scale=0.01, size=self.rf.shape)
        )

        self.sparse_input = []
        self.dense_input = np.zeros(self.ff_size, dtype=int)

        self.n_computes = 1
        self.feedforward_trace = np.full(self.ff_size, self.feedforward_sds.sparsity)
        self.output_trace = np.full(self.output_size, self.output_sds.sparsity)
        self.recognition_strength_trace = 0

        self.base_boosting_k = boosting_k
        self.newborn_pruning_cycle = newborn_pruning_cycle
        self.newborn_pruning_stages = newborn_pruning_stages
        self.newborn_pruning_mode = newborn_pruning_mode
        self.newborn_pruning_stage = 0
        self.prune_grow_cycle = prune_grow_cycle

        self.run_time = 0

    def compute(self, input_sdr: SparseSdr, learn: bool = False) -> SparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.run_time += run_time
        return output_sdr

    @timed
    def _compute(self, input_sdr: SparseSdr, learn: bool) -> SparseSdr:
        self.n_computes += 1
        self.feedforward_trace[input_sdr] += 1

        if self.is_newborn_phase:
            if self.n_computes % int(self.newborn_pruning_cycle * self.output_size) == 0:
                self.shrink_receptive_field()
        else:
            if self.n_computes % int(self.prune_grow_cycle * self.output_size) == 0:
                self.prune_grow_synapses()

        self.update_input(input_sdr)

        match_mask = self.match_input(self.dense_input)
        overlaps = (match_mask * self.weights).sum(axis=1)

        if self.is_newborn_phase:
            # boosting
            boosting_alpha = boosting(relative_rate=self.output_relative_rate, k=self.boosting_k)
            # ^ sign(B) is to make boosting direction unaffected by the sign of the overlap
            overlaps = overlaps * boosting_alpha ** np.sign(overlaps)

        n_winners = self.output_sds.active_size
        winners = np.sort(
            np.argpartition(-overlaps, n_winners)[:n_winners]
        )
        winners = winners[overlaps[winners] > 0]

        # update winners activation stats
        self.output_trace[winners] += 1
        self.recognition_strength_trace += overlaps[winners].sum() / n_winners

        if learn:
            self.learn(winners, match_mask[winners])
        return winners

    def learn(self, neurons: np.ndarray, match_input_mask: np.ndarray, modulation: float = 1.0):
        if len(neurons) == 0:
            return

        w = self.weights[neurons]
        lr = modulation * self.learning_rate
        mask = match_input_mask
        matched = mask.sum(axis=1, keepdims=True)

        empty_hits = matched.flatten() == 0
        if np.any(empty_hits):
            non_empty_hits = np.logical_not(empty_hits)
            w = w[non_empty_hits]
            mask = mask[non_empty_hits]
            matched = matched[non_empty_hits]

        dw_inh = lr * (1 - mask) * w
        dw_pool = dw_inh.sum(axis=1, keepdims=True)
        dw_exc = mask * dw_pool / matched
        # dw_exc = lr * mask * w

        self.weights[neurons] = self.normalize_weights(w + dw_exc - dw_inh)

    def process_feedback(self, feedback_sdr: SparseSdr):
        # feedback SDR is the SP neurons that should be reinforced
        fb_match_mask, _ = self.match_input(self.dense_input, neurons=feedback_sdr)
        self.learn(feedback_sdr, fb_match_mask)

    def shrink_receptive_field(self):
        self.newborn_pruning_stage += 1

        if self.newborn_pruning_mode == 'linear':
            new_sparsity = self.current_rf_sparsity_linear()
        elif self.newborn_pruning_mode == 'powerlaw':
            new_sparsity = self.current_rf_sparsity_powerlaw()
        else:
            raise ValueError(f'Pruning mode {self.newborn_pruning_mode} is not supported')
        if new_sparsity > self.rf_sparsity:
            # if feedforward sparsity is tracked, then it may change and lead to RF increase
            # ==> leave RF as is
            return

        # probabilities to keep connection
        keep_prob = np.power(np.abs(self.weights), 2.0) + 0.01
        keep_prob /= keep_prob.sum(axis=1, keepdims=True)

        # sample what connections to keep for each neuron independently
        new_rf_size = round(new_sparsity * self.ff_size)
        keep_connections_i = sample_for_each_neuron(
            rng=self.rng, n_neurons=self.output_size,
            set_size=self.rf_size, sample_size=new_rf_size, probs_2d=keep_prob
        )

        self.rf = gather_rows(self.rf, keep_connections_i)
        self.weights = self.normalize_weights(
            gather_rows(self.weights, keep_connections_i)
        )
        print(f'Prune newborns: {self._state_str()}')

        if not self.is_newborn_phase:
            # it is ended
            self.on_end_newborn_phase()

        self.prune_grow_synapses()

    def prune_grow_synapses(self):
        print(f'Force neurogenesis: {self.output_entropy():.3f} | {self.recognition_strength:.1f}')
        # FIXME: rework prune/grow
        return
        # prune-grow operation combined results to resample of a part of
        # the most inactive or just randomly selected synapses;
        # new synapses are distributed according to the feedforward distribution
        synapse_sample_prob = self.feedforward_rate
        synapse_sample_prob /= synapse_sample_prob.sum()

        for neuron in range(self.output_size):
            if self.output_relative_rate[neuron] > .1:
                continue

            self.rf[neuron] = self.rng.choice(
                self.ff_size, size=self.rf_size, replace=False,
                p=synapse_sample_prob
            )
            self.weights[neuron] = 1 / self.rf_size

    def on_end_newborn_phase(self):
        self.learning_rate /= 2
        print(f'Become adult: {self._state_str()}')

    def update_input(self, sdr: SparseSdr):
        # erase prev SDR
        self.dense_input[self.sparse_input] = 0
        # set new SDR
        self.sparse_input = sdr
        self.dense_input[self.sparse_input] = 1

    def match_input(self, dense_input, neurons: np.ndarray = None):
        rf = self.rf if neurons is None else self.rf[neurons]
        return dense_input[rf]

    @staticmethod
    def normalize_weights(weights):
        normalizer = np.abs(weights).sum(axis=1, keepdims=True)
        return np.clip(weights / normalizer, 0., 1)

    def get_active_rf(self, weights):
        w_thr = 1 / self.rf_size
        return weights >= w_thr

    def current_rf_sparsity_linear(self):
        ff_sparsity = (
            self.ff_avg_sparsity if self.adapt_to_ff_sparsity else self.feedforward_sds.sparsity
        )
        final_rf_sparsity = min(
            self.max_rf_sparsity,
            self.max_rf_to_input_ratio * ff_sparsity
        )

        newborn_phase_progress = self.newborn_pruning_stage / self.newborn_pruning_stages
        initial, final = self.initial_rf_sparsity, final_rf_sparsity
        return initial + newborn_phase_progress * (final - initial)

    def current_rf_sparsity_powerlaw(self):
        ff_sparsity = (
            self.ff_avg_sparsity if self.adapt_to_ff_sparsity else self.feedforward_sds.sparsity
        )
        final_rf_sparsity = min(
            self.max_rf_sparsity,
            self.max_rf_to_input_ratio * ff_sparsity
        )

        steps_left = self.newborn_pruning_stages - self.newborn_pruning_stage + 1
        current_rf_sparsity = self.rf_sparsity
        rf_decay = np.power(final_rf_sparsity / current_rf_sparsity, 1 / steps_left)
        return current_rf_sparsity * rf_decay

    @property
    def ff_size(self):
        return self.feedforward_sds.size

    @property
    def ff_avg_active_size(self):
        return self.feedforward_trace.sum() // self.n_computes

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
    def n_groups(self):
        return self.output_sds.active_size

    @property
    def output_size(self):
        return self.output_sds.size

    @property
    def is_newborn_phase(self):
        return self.newborn_pruning_stage < self.newborn_pruning_stages

    def _state_str(self) -> str:
        return f'{self.rf_sparsity:.4f} | {self.rf_size} | {self.learning_rate:.3f}' \
               f' | {self.boosting_k:.2f}'

    @property
    def feedforward_rate(self):
        return self.feedforward_trace / self.n_computes

    @property
    def output_rate(self):
        return self.output_trace / self.n_computes

    @property
    def output_relative_rate(self):
        target_rate = self.output_sds.sparsity
        return self.output_rate / target_rate

    @property
    def rf_match_trace(self):
        return self.feedforward_trace[self.rf]

    @property
    def boosting_k(self):
        if not self.is_newborn_phase:
            return 0.
        newborn_phase_progress = self.newborn_pruning_stage / self.newborn_pruning_stages
        return self.base_boosting_k * (1 - newborn_phase_progress)

    def output_entropy(self):
        return entropy(self.output_rate, sds=self.output_sds)

    @property
    def recognition_strength(self):
        return self.recognition_strength_trace / self.n_computes