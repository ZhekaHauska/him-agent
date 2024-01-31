#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from enum import auto, Enum

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


class SpNewbornPruningMode(Enum):
    LINEAR = 1
    POWERLAW = auto()


# DEPRECATED: old version of the SpatialPooler to check against the
# new generalized float/binary-SP. For it: sp_float is the first stable version
# and sp_rate is planned to be the dev extension.
# It is scheduled for removal soon after the latter is stabilized/tested.


class SpatialPooler:
    rng: Generator

    # I/O settings
    feedforward_sds: Sds
    adapt_to_ff_sparsity: bool

    output_sds: Sds

    initial_rf_sparsity: float
    target_max_rf_sparsity: float
    target_rf_to_input_ratio: float
    rf: np.ndarray
    weights: np.ndarray

    # newborn stage
    newborn_pruning_mode: SpNewbornPruningMode
    newborn_pruning_cycle: float
    newborn_pruning_stages: int
    newborn_pruning_schedule: int
    newborn_pruning_stage: int
    prune_grow_cycle: float
    #   boosting. It is active only during newborn stage
    base_boosting_k: float
    boosting_k: float

    initial_learning_rate: float
    learning_rate: float

    # cache
    sparse_input: SparseSdr
    dense_input: np.ndarray

    winners: SparseSdr
    potentials: np.ndarray

    # stats
    n_computes: int
    run_time: float
    #   input values accumulator
    feedforward_trace: np.ndarray
    #   input size accumulator
    feedforward_size_trace: float
    #   output values accumulator
    output_trace: np.ndarray
    #   recognition strength is an avg winners' overlap (potential)
    recognition_strength_trace: float

    def __init__(
            self, *, seed: int,
            feedforward_sds: Sds,
            adapt_to_ff_sparsity: bool,
            # newborn / mature
            initial_max_rf_sparsity: float, target_max_rf_sparsity: float,
            initial_rf_to_input_ratio: float, target_rf_to_input_ratio: float,
            output_sds: Sds,
            learning_rate: float,
            # neurogenesis
            newborn_pruning_mode: str,
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            prune_grow_cycle: float, boosting_k: float,
    ):
        self.rng = np.random.default_rng(seed)

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.adapt_to_ff_sparsity = adapt_to_ff_sparsity

        self.output_sds = Sds.make(output_sds)

        self.initial_rf_sparsity = min(
            initial_rf_to_input_ratio * self.feedforward_sds.sparsity,
            initial_max_rf_sparsity
        )
        self.target_rf_to_input_ratio = target_rf_to_input_ratio
        self.target_max_rf_sparsity = target_max_rf_sparsity

        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate

        rf_size = int(self.initial_rf_sparsity * self.ff_size)
        self.rf = sample_for_each_neuron(
            rng=self.rng, n_neurons=self.output_size,
            set_size=self.ff_size, sample_size=rf_size
        )
        print(f'SP.vec weights init shape: {self.rf.shape}')
        self.weights = self.normalize_weights(
            self.rng.normal(loc=1.0, scale=0.0001, size=self.rf.shape)
        )

        self.base_boosting_k = boosting_k
        self.boosting_k = self.base_boosting_k
        self.newborn_pruning_cycle = newborn_pruning_cycle
        self.newborn_pruning_schedule = int(self.newborn_pruning_cycle / self.output_sds.sparsity)
        self.newborn_pruning_stages = newborn_pruning_stages
        self.newborn_pruning_mode = SpNewbornPruningMode[newborn_pruning_mode.upper()]
        self.newborn_pruning_stage = 0
        self.prune_grow_cycle = prune_grow_cycle
        self.prune_grow_schedule = int(self.prune_grow_cycle / self.output_sds.sparsity)

        self.sparse_input = []
        # use float to eliminate inevitable int-to-float converting when we multiply it by weights
        self.dense_input = np.zeros(self.ff_size, dtype=float)
        self.winners = []
        self.potentials = np.zeros(self.output_size)

        self.n_computes = 1
        self.feedforward_trace = np.full(self.ff_size, self.feedforward_sds.sparsity)
        self.feedforward_size_trace = 0.
        self.output_trace = np.full(self.output_size, self.output_sds.sparsity)
        self.recognition_strength_trace = 0
        self.run_time = 0

    def compute(self, input_sdr: SparseSdr, learn: bool = False) -> SparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.run_time += run_time
        return output_sdr

    @timed
    def _compute(self, input_sdr: SparseSdr, learn: bool) -> SparseSdr:
        self.accept_input(input_sdr, learn=learn)
        self.try_activate_neurogenesis()

        matched_input_activity = self.match_current_input()
        delta_potentials = (matched_input_activity * self.weights).sum(axis=1)
        self.potentials += self.apply_boosting(delta_potentials)

        # move most to the compute winners
        self.select_winners()
        self.reinforce_winners(matched_input_activity, learn)

        output_sdr = self.select_output()
        self.accept_output(output_sdr, learn=learn)

        return output_sdr

    def accept_input(self, sdr: SparseSdr, *, learn: bool):
        """Accept new input and move to the next time step"""
        # forget prev SDR
        self.dense_input[self.sparse_input] = 0.
        # apply timed decay to neurons' potential
        self.potentials.fill(0.)

        # set new SDR
        self.sparse_input = sdr
        self.dense_input[self.sparse_input] = 1.0

        # For SP, an online learning is THE MOST natural operation mode.
        # We treat the opposite case as the special mode, which only partly affects SP state.
        if learn:
            self.n_computes += 1
            self.feedforward_trace[self.sparse_input] += 1.0
            self.feedforward_size_trace += len(self.sparse_input)

    def try_activate_neurogenesis(self):
        if self.is_newborn_phase:
            if self.n_computes % self.newborn_pruning_schedule == 0:
                self.shrink_receptive_field()
        else:
            if self.n_computes % self.prune_grow_schedule == 0:
                self.prune_grow_synapses()

    def match_current_input(self, with_neurons: np.ndarray = None):
        rf = self.rf if with_neurons is None else self.rf[with_neurons]
        return self.dense_input[rf]

    def apply_boosting(self, overlaps):
        if self.is_newborn_phase and self.boosting_k > 1e-2:
            # boosting
            boosting_alpha = boosting(relative_rate=self.output_relative_rate, k=self.boosting_k)
            # FIXME: normalize boosting alpha over neurons
            overlaps = overlaps * boosting_alpha
        return overlaps

    def select_winners(self):
        n_winners = self.output_sds.active_size
        winners = np.sort(
            np.argpartition(self.potentials, -n_winners)[-n_winners:]
        )
        self.winners = winners[self.potentials[winners] > 0]

    def reinforce_winners(self, matched_input_activity, learn: bool):
        if not learn:
            return
        self.stdp(self.winners, matched_input_activity[self.winners])

    def stdp(
            self, neurons: SparseSdr, rf_match_input_mask: np.ndarray,
            modulation: float = 1.0
    ):
        if len(neurons) == 0:
            return

        w = self.weights[neurons]
        n_matched = rf_match_input_mask.sum(axis=1, keepdims=True) + .1
        lr = modulation * self.learning_rate / n_matched

        dw_matched = rf_match_input_mask * lr

        self.weights[neurons] = self.normalize_weights(w + dw_matched)

    def select_output(self):
        output_sdr = self.winners
        return output_sdr

    def accept_output(self, sdr: SparseSdr, *, learn: bool):
        if not learn or sdr.shape[0] <= 0:
            return

        # update winners activation stats
        self.output_trace[sdr] += 1.0
        self.recognition_strength_trace += self.potentials[sdr].mean()

    def get_step_debug_info(self):
        return {
            'potentials': np.sort(self.potentials),
            'recognition_strength': self.potentials[self.winners],
            'weights': self.weights,
            'rf': self.rf,
        }

    def process_feedback(self, feedback_sdr: SparseSdr, modulation: float = 1.0):
        # feedback SDR is the SP neurons that should be reinforced or punished
        fb_match_mask = self.match_current_input(with_neurons=feedback_sdr)
        self.stdp(feedback_sdr, fb_match_mask, modulation=modulation)

    def shrink_receptive_field(self):
        self.newborn_pruning_stage += 1

        if self.newborn_pruning_mode == 'linear':
            new_sparsity = self.newborn_linear_progress(
                initial=self.initial_rf_sparsity, target=self.get_target_rf_sparsity()
            )
        elif self.newborn_pruning_mode == 'powerlaw':
            new_sparsity = self.newborn_powerlaw_progress(
                current=self.rf_sparsity, target=self.get_target_rf_sparsity()
            )
        else:
            raise ValueError(f'Pruning mode {self.newborn_pruning_mode} is not supported')

        if new_sparsity > self.rf_sparsity:
            # if feedforward sparsity is tracked, then it may change and lead to RF increase
            # ==> leave RF as is
            return

        # probabilities to keep connection
        threshold = .5 / self.rf_size
        keep_prob = np.power(np.abs(self.weights) / threshold + 0.1, 2.0)
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
        self.learning_rate = self.newborn_linear_progress(
            initial=self.initial_learning_rate, target=0.2 * self.initial_learning_rate
        )
        self.boosting_k = self.newborn_linear_progress(
            initial=self.base_boosting_k, target=0.
        )
        print(f'Prune newborns: {self._state_str()}')

        if not self.is_newborn_phase:
            # it is ended
            self.on_end_newborn_phase()

        self.prune_grow_synapses()

    def prune_grow_synapses(self):
        # FIXME: implement prune/grow
        print(
            f'{self.output_entropy():.3f}'
            f' | {self.recognition_strength:.1f}'
        )

    def on_end_newborn_phase(self):
        # self.learning_rate /= 2
        self.boosting_k = 0.
        print(f'Become adult: {self._state_str()}')

    @staticmethod
    def normalize_weights(weights):
        normalizer = np.abs(weights).sum(axis=1, keepdims=True)
        return np.clip(weights / normalizer, 0., 1)

    def get_active_rf(self, weights):
        w_thr = 1 / self.rf_size
        return weights >= w_thr

    def get_target_rf_sparsity(self):
        ff_sparsity = (
            self.ff_avg_sparsity if self.adapt_to_ff_sparsity else self.feedforward_sds.sparsity
        )
        return min(
            self.target_rf_to_input_ratio * ff_sparsity,
            self.target_max_rf_sparsity,
        )

    def newborn_linear_progress(self, initial, target):
        newborn_phase_progress = self.newborn_pruning_stage / self.newborn_pruning_stages
        return initial + newborn_phase_progress * (target - initial)

    def newborn_powerlaw_progress(self, current, target):
        steps_left = self.newborn_pruning_stages - self.newborn_pruning_stage + 1
        current = self.rf_sparsity
        decay = np.power(target / current, 1 / steps_left)
        return current * decay

    @property
    def ff_size(self):
        return self.feedforward_sds.size

    @property
    def ff_avg_active_size(self):
        return self.feedforward_size_trace // self.n_computes

    @property
    def ff_avg_sparsity(self):
        return self.ff_avg_active_size / self.ff_size

    @property
    def rf_size(self) -> int:
        return self.rf.shape[1]

    @property
    def rf_sparsity(self):
        return self.rf_size / self.ff_size

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

    def output_entropy(self):
        return entropy(self.output_rate, sds=self.output_sds)

    @property
    def recognition_strength(self):
        return self.recognition_strength_trace / self.n_computes
