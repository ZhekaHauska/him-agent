#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

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


class SpatialTemporalPooler:
    rng: Generator

    # input
    feedforward_sds: Sds
    feedforward_trace: np.ndarray

    # connections
    rf: np.ndarray
    weights: np.ndarray
    temporal_decay: float | np.ndarray
    temporal_decay_mean: float
    overlap_trace: np.ndarray
    reset_on_activation: bool

    # output
    output_sds: Sds
    min_overlap_for_activation: float
    n_computes: int
    output_trace: np.ndarray

    # learning
    learning_rate: float
    global_inhibition_strength: float

    # Newborn stage
    base_boosting_k: float
    initial_rf_sparsity: float
    max_rf_sparsity: float
    max_rf_to_input_ratio: float
    # Newborn pruning
    newborn_pruning_cycle: float
    newborn_pruning_stages: int
    newborn_pruning_stage: int
    # Adult pruning/growing/resampling
    prune_grow_cycle: float

    # auxiliary
    sparse_input: SparseSdr
    dense_input: np.ndarray
    recognition_strength_trace: float
    run_time: float

    def __init__(
            self, feedforward_sds: Sds,
            pooling_window: float,
            # newborn / mature
            initial_rf_to_input_ratio: float, max_rf_to_input_ratio: float, max_rf_sparsity: float,
            output_sds: Sds,
            min_overlap_for_activation: float,
            learning_rate: float, global_inhibition_strength: float,
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            prune_grow_cycle: float,
            boosting_k: float, seed: int,
            adapt_to_ff_sparsity: bool = True,
            rand_decay_max_ratio: float = 1.,
            reset_on_activation: bool = True,
    ):
        self.rng = np.random.default_rng(seed)
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds

        self.adapt_to_ff_sparsity = adapt_to_ff_sparsity

        self.initial_rf_sparsity = min(
            initial_rf_to_input_ratio * self.feedforward_sds.sparsity,
            0.65
        )
        self.max_rf_to_input_ratio = max_rf_to_input_ratio
        self.max_rf_sparsity = max_rf_sparsity

        self.min_overlap_for_activation = min_overlap_for_activation
        self.learning_rate = learning_rate
        self.global_inhibition_strength = global_inhibition_strength
        self.polarity = 1

        rf_size = int(self.initial_rf_sparsity * self.ff_size)
        self.rf = sample_for_each_neuron(
            rng=self.rng, n_neurons=self.output_size,
            set_size=self.ff_size, sample_size=rf_size
        )
        print(f'SP vec init shape: {self.rf.shape}')

        w0 = 1 / rf_size
        self.w_min = 0.
        self.weights = self.normalize_weights(
            self.rng.normal(w0, w0, size=self.rf.shape)
        )

        self.sparse_input = []
        self.dense_input = np.zeros(self.ff_size)

        self.n_computes = 0
        self.feedforward_trace = np.full(self.ff_size, 1e-5)
        self.output_trace = np.full(self.output_size, 1e-5)
        self.recognition_strength_trace = 0

        # temporal decay
        self.overlap_trace = np.zeros(self.output_size)
        assert rand_decay_max_ratio >= 1., 'Decay max ratio must be >= 1.0'
        self.temporal_decay = _make_decay_matrix(
            rng=self.rng, size=self.output_size,
            temporal_window_mean=pooling_window,
            temporal_window_max_ratio=rand_decay_max_ratio
        )
        self.match_mask_trace = np.zeros_like(self.rf)
        self.reset_on_activation = reset_on_activation

        self.base_boosting_k = boosting_k
        self.newborn_pruning_cycle = newborn_pruning_cycle
        self.newborn_pruning_stages = newborn_pruning_stages
        self.newborn_pruning_stage = 0
        self.prune_grow_cycle = prune_grow_cycle

        self.no_feedback_count = 0
        self.run_time = 0

    def compute(self, input_sdr: SparseSdr, learn: bool = False) -> SparseSdr:
        """Compute the output SDR."""
        # TODO: rename to feedforward
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.run_time += run_time
        return output_sdr

    @timed
    def _compute(self, input_sdr: SparseSdr, learn: bool) -> SparseSdr:
        self.n_computes += 1
        self.no_feedback_count += 1
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

        # print(self.match_mask_trace.shape, self.temporal_decay.shape, match_mask.shape)
        self.match_mask_trace = (
                self.match_mask_trace * self.temporal_decay[..., np.newaxis] + match_mask
        )
        self.overlap_trace = self.overlap_trace * self.temporal_decay + overlaps

        n_winners = self.output_sds.active_size
        winners = np.sort(
            np.argpartition(-self.overlap_trace, n_winners)[:n_winners]
        )

        # update winners activation stats
        self.output_trace[winners] += 1
        self.recognition_strength_trace += (
                self.overlap_trace[winners].sum() / n_winners / self.rf_size
        )

        if learn:
            self.learn(winners)

        # reset overlap and match mask traces for winners
        if self.reset_on_activation:
            self.match_mask_trace[winners] = 0.
            self.overlap_trace[winners] = 0.

        return winners

    def learn(self, neurons: np.ndarray, modulation: float = 1.0):
        w = self.weights[neurons]
        match_trace = self.match_mask_trace[neurons]
        matched = match_trace.sum(axis=1, keepdims=True)
        # add anti zero-division term
        matched = matched + (matched == 0.) * 1e-5

        lr = modulation * self.polarity * self.learning_rate
        inh = self.global_inhibition_strength
        dw_inh = lr * inh * (1 - match_trace)

        # inhibition frees weights to be used for reinforcement
        dw_pool = dw_inh.sum(axis=1, keepdims=True)
        dw_exc = match_trace * dw_pool / matched

        self.weights[neurons] = self.normalize_weights(w + dw_exc - dw_inh)

    def process_feedback(self, feedback_sdr: SparseSdr):
        assert False, 'Fix feedback processing as match mask is incorrect now'
        # feedback SDR is the SP neurons that should be reinforced
        feedback_strength = self.no_feedback_count
        fb_match_mask, _ = self.match_input(self.dense_input, neurons=feedback_sdr)

        self.learn(feedback_sdr, fb_match_mask, modulation=feedback_strength)
        self.no_feedback_count = 0

    def shrink_receptive_field(self):
        self.newborn_pruning_stage += 1

        new_sparsity = self.current_rf_sparsity()
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
        self.match_mask_trace = gather_rows(self.match_mask_trace, keep_connections_i)
        print(f'Prune newborns: {self._state_str()}')

        if not self.is_newborn_phase:
            # it is ended
            self.on_end_newborn_phase()

        self.prune_grow_synapses()

    def prune_grow_synapses(self):
        print(f'Force neurogenesis: {self.output_entropy():.3f} | {self.recognition_strength:.1f}')
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

    def normalize_weights(self, weights):
        return np.clip(
            weights / np.abs(weights).sum(axis=1, keepdims=True),
            self.w_min, 1
        )

    def get_active_rf(self, weights):
        w_thr = 1 / self.rf_size
        return weights >= w_thr

    def current_rf_sparsity(self):
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


def _resolve_temporal_decay(temporal_window: float = None) -> float:
    if 0.0 <= temporal_window < 1.0:
        decay = temporal_window
        return decay
    elif temporal_window >= 1.0:
        # temporal window: decay ^ window = 1 / 10 ==> ten-time decrease
        base = 0.1
        return np.power(base, 1 / temporal_window)

    raise ValueError(f'Invalid temporal window value {temporal_window}!')


def _make_decay_matrix(rng, size, temporal_window_mean, temporal_window_max_ratio):
    if temporal_window_max_ratio == 1.0:
        scales = np.ones(size)
    else:
        scales = _loguniform(rng, temporal_window_max_ratio, size=size)

    temporal_window_mx = temporal_window_mean * scales
    temporal_window_mx[temporal_window_mx <= 1.0] = 0.
    base = 0.1
    temporal_window_mx[temporal_window_mx > 1.0] = np.power(
        base,
        1. / temporal_window_mx[temporal_window_mx > 1.0]
    )
    return temporal_window_mx


def _loguniform(rng, std, size):
    low = np.log(1 / std)
    high = np.log(std)
    return np.exp(rng.uniform(low, high, size))
