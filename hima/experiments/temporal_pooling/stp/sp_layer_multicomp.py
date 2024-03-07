#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from enum import Enum, auto
from typing import cast

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sdrr import (
    RateSdr, AnySparseSdr, OutputMode, split_sdr_values,
    CompartmentsAnySparseSdr
)
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.sp import SpNewbornPruningMode
from hima.experiments.temporal_pooling.stp.sp_utils import (
    boosting, RepeatingCountdown, make_repeating_counter
)


# Intermediate version of SP during the migration to work with both
# binary and rate encodings. Previous: sp_rate.py. Current last.
# New: supports synaptogenesis.

class SpLearningAlgo(Enum):
    OLD = 1
    NEW = auto()
    NEW_SQ = auto()

# TODO:
#   - [ ] SdrCache (dense+sparse datastruct)


class SpatialPooler:
    """A competitive network (as meant by Rolls)."""
    rng: Generator

    compartments: dict
    compartments_ix: dict[str, int]
    compartments_weight: npt.NDArray[float]

    # I/O settings
    # feedforward_sds: Sds
    # adapt_to_ff_sparsity: bool

    output_sds: Sds
    output_mode: OutputMode

    # initial_rf_sparsity: float
    # target_max_rf_sparsity: float
    # target_rf_to_input_ratio: float
    # rf: npt.NDArray[int]
    # weights: npt.NDArray[float]

    neurogenesis_countdown: RepeatingCountdown
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

    sample_winners_frac: float
    winners: SparseSdr
    strongest_winner: int | None
    potentials: np.ndarray

    # stats
    #   average computation time
    computation_speed: MeanValue[float]
    #   output values accumulator
    slow_output_trace: MeanValue[npt.NDArray[float]]
    #   recognition strength is an avg winners' overlap (potential)
    slow_recognition_strength_trace: MeanValue[float]

    def __init__(
            self, *, global_config, seed: int,
            compartments: list[str], compartments_config: dict[str, dict],
            compartments_weight: dict[str, float],
            # output
            output_sds: Sds, output_mode: str,
            # learning
            learning_rate: float, learning_algo: str,
            # newborn phase
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            newborn_pruning_mode: str, boosting_k: float,
            # mature phase
            prune_grow_cycle: float,
            # additional optional params
            normalize_rates: bool = True, sample_winners: float | None = None,
            # rand_weights_ratio: 0.7, rand_weights_noise: 0.01,
            slow_trace_decay: float = 0.99, fast_trace_decay: float = 0.95,
    ):
        self.rng = np.random.default_rng(seed)
        print(f'{output_sds=}')

        if global_config is None:
            ...
        else:
            from hima.common.config.global_config import GlobalConfig
            global_config: GlobalConfig
            self.compartments = {
                name: global_config.resolve_object(
                    compartments_config[name], output_sds=output_sds
                )
                for name in compartments
            }

        self.compartments_ix = {
            name: i
            for i, name in enumerate(compartments)
        }
        self.compartments_weight = np.array([
            compartments_weight[name] for name in compartments
        ])
        self.compartments_weight /= self.compartments_weight.sum()
        for comp_name in self.compartments:
            newborn_pruning_cycle = self.compartments[comp_name].newborn_pruning_cycle
            newborn_pruning_stages = self.compartments[comp_name].newborn_pruning_stages

        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode[output_mode.upper()]
        self.normalize_rates = normalize_rates

        self.learning_algo = SpLearningAlgo[learning_algo.upper()]

        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.modulation = 1.0

        if sample_winners is None or not sample_winners:
            self.select_winners = self._select_winners
        else:
            self.select_winners = self._sample_winners
            self.sample_winners_frac = sample_winners

        self.newborn_pruning_mode = SpNewbornPruningMode[newborn_pruning_mode.upper()]
        self.newborn_pruning_cycle = newborn_pruning_cycle
        self.newborn_pruning_schedule = int(self.newborn_pruning_cycle / self.output_sds.sparsity)
        self.newborn_pruning_stages = newborn_pruning_stages
        self.newborn_pruning_stage = 0
        self.prune_grow_cycle = prune_grow_cycle
        self.prune_grow_schedule = int(self.prune_grow_cycle / self.output_sds.sparsity)
        self.base_boosting_k = boosting_k
        self.boosting_k = self.base_boosting_k
        self.neurogenesis_countdown = make_repeating_counter(self.newborn_pruning_schedule)

        if not self.is_newborn_phase:
            self.on_end_newborn_phase()

        self.winners = np.empty(0, dtype=int)
        self.winners_value = 1.0
        self.strongest_winner = None
        self.potentials = np.zeros(self.output_size)

        # stats collection
        self.health_check_results = {}
        self.slow_health_check_results = {}

        self.computation_speed = MeanValue(exp_decay=slow_trace_decay)
        self.slow_recognition_strength_trace = MeanValue(exp_decay=slow_trace_decay)
        self.slow_output_trace = MeanValue(
            size=self.output_size, exp_decay=slow_trace_decay
        )
        self.slow_output_trace.put(self.output_sds.sparsity)
        self.fast_output_trace = MeanValue(
            size=self.output_size, exp_decay=fast_trace_decay
        )
        self.fast_output_trace.put(self.output_sds.sparsity)

        self.neurogenesis_cnt = 0
        self.neurogenesis_queue = np.zeros(self.output_size, dtype=float)

        # synchronize selected attributes for compartments with the main SP
        for comp_name in self.compartments:
            self.compartments[comp_name].output_mode = self.output_mode

    def compute(self, input_sdr: CompartmentsAnySparseSdr, learn: bool = False) -> AnySparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.computation_speed.put(run_time)
        return output_sdr

    @timed
    def _compute(self, input_sdr: CompartmentsAnySparseSdr, learn: bool) -> AnySparseSdr:
        assert len(input_sdr) == len(self.compartments), (
            f'{input_sdr=} | {self.compartments.keys()=}'
        )

        matched_input_activity = self.match_input(input_sdr, learn)
        self.select_winners()
        self.broadcast_winners()
        self.reinforce_winners(matched_input_activity, learn)

        output_sdr = self.select_output()
        self.accept_output(output_sdr, learn=learn)
        return output_sdr

    def match_input(self, input_sdr: CompartmentsAnySparseSdr, learn):
        matched_input_activity = {}
        delta_potentials = np.zeros_like(self.potentials)

        for comp_name in self.compartments:
            compartment = self.compartments[comp_name]
            compartment_weight = self.compartments_weight[self.compartments_ix[comp_name]]

            matched_input_activity[comp_name] = compartment.match_input(
                input_sdr[comp_name], learn=learn
            )
            # NB: compartment potentials contain only current time step induced potentials
            delta_potentials += compartment_weight * compartment.potentials

        # NB: boosting is a neuron-level effect, thus we apply it here
        self.potentials += self.apply_boosting(delta_potentials)
        return matched_input_activity

    def broadcast_winners(self):
        for comp_name in self.compartments:
            self.compartments[comp_name].accept_winners(
                winners=self.winners, winners_value=self.winners_value,
                strongest_winner=self.strongest_winner
            )

    def reinforce_winners(self, matched_input_activity, learn: bool):
        if not learn:
            return

        for comp_name in self.compartments:
            self.compartments[comp_name].reinforce_winners(
                matched_input_activity[comp_name], learn
            )

    def get_step_debug_info(self):
        return {
            'potentials': np.sort(self.potentials),
            'recognition_strength': self.potentials[self.winners],
            # 'weights': self.weights,
            # 'rf': self.rf
        }

    def try_activate_neurogenesis(self, learn):
        raise NotImplementedError('Not implemented yet')

    def apply_boosting(self, overlaps):
        if self.is_newborn_phase and self.boosting_k > 1e-2:
            # boosting
            boosting_alpha = boosting(relative_rate=self.output_relative_rate, k=self.boosting_k)
            # FIXME: normalize boosting alpha over neurons
            overlaps = overlaps * boosting_alpha
        return overlaps

    def _select_winners(self):
        n_winners = self.output_sds.active_size

        winners = np.argpartition(self.potentials, -n_winners)[-n_winners:]
        self.strongest_winner = cast(int, winners[np.argmax(self.potentials[winners])])
        winners.sort()

        self.winners = winners[self.potentials[winners] > 0]
        if self.output_mode == OutputMode.RATE:
            self.winners_value = self.potentials[self.winners].copy()
            if self.normalize_rates:
                self.winners_value = safe_divide(
                    self.winners_value,
                    cast(float, self.potentials[self.strongest_winner])
                )

    def _sample_winners(self):
        n_winners = self.output_sds.active_size

        n_pot_winners = min(round(self.sample_winners_frac*n_winners), self.potentials.size)
        pot_winners = np.argpartition(self.potentials, -n_pot_winners)[-n_pot_winners:]
        self.strongest_winner = cast(int, pot_winners[np.argmax(self.potentials[pot_winners])])

        probs = self.potentials[pot_winners]**2 + 1e-10
        norm = np.sum(probs)

        winners = self.rng.choice(
            pot_winners, size=n_winners, replace=False,
            p=safe_divide(probs, norm)
        )
        winners.sort()

        self.winners = winners[self.potentials[winners] > 0]
        if self.output_mode == OutputMode.RATE:
            self.winners_value = self.potentials[self.winners].copy()
            if self.normalize_rates:
                self.winners_value = safe_divide(
                    self.winners_value,
                    cast(float, self.potentials[self.strongest_winner])
                )

    def select_output(self):
        if self.output_mode == OutputMode.RATE:
            return RateSdr(self.winners, values=self.winners_value)
        return self.winners

    def accept_output(self, sdr: SparseSdr, *, learn: bool):
        for comp_name in self.compartments:
            self.compartments[comp_name].accept_output(sdr, learn=learn)

        sdr, value = split_sdr_values(sdr)
        if not learn or sdr.shape[0] == 0:
            return

        # update winners activation stats
        self.slow_output_trace.put(value, sdr)
        # FIXME: make two metrics: for pre-weighting, post weighting delta
        self.slow_recognition_strength_trace.put(
            self.potentials[sdr].mean()
        )

        self.fast_output_trace.put(value, sdr)

    def process_feedback(self, feedback_sdr: SparseSdr, modulation: float = 1.0):
        raise NotImplementedError('Not implemented yet')

    def prune_grow_synapses(self):
        # NB: usually we work in log-space => log_ prefix is mostly omit for vars
        self.health_check()

        # self.rng.binomial(1, p=self.neurogenesis_queue)
        self.neurogenesis_queue[:] = 0.
        rnd_cnt = 0

        # TODO: optimize thresholds and power scaling;
        #   also remember cumulative effect of prob during the cycle for frequent neurons
        abs_rate_low, abs_rate_high = np.log(20), np.log(100)
        eff_low, eff_high = np.log(1.5), np.log(20)

        # Step 1: deal with absolute rates: sleepy input RF and output
        # TODO: consider increasing random potential part for them
        rnd_cnt += self.apply_synaptogenesis_to_metric(
            self.slow_health_check_results['ln(nrfe_in)'],
            abs_rate_low, abs_rate_high,
            prob_power=2.0,
            apply_random_synaptogenesis=True
        )
        rnd_cnt += self.apply_synaptogenesis_to_metric(
            self.health_check_results['ln(op)'],
            abs_rate_low, abs_rate_high,
            prob_power=2.0,
            apply_random_synaptogenesis=True
        )

        # Step 2: deal with sleepy output
        rnd_cnt += self.apply_synaptogenesis_to_metric(
            self.health_check_results['ln(nrfe_out)'],
            eff_low, eff_high,
            prob_power=1.5,
        )

        print(
            f'{self.output_entropy():.3f}'
            f' | {self.recognition_strength:.1f}'
            f' | {self.health_check_results["avg(rfe_out)"]:.3f}'
            f' | {self.neurogenesis_cnt}'
            f' | {rnd_cnt}'
            f' | {np.sum(self.neurogenesis_queue):.2f}'
        )

        self.neurogenesis_cnt = 0
        self.decay_stat_trackers()

    def apply_synaptogenesis_to_metric(
            self, metric: npt.NDArray[float], low: float, high: float,
            prob_power: float = 1.5,
            apply_random_synaptogenesis: bool = False
    ):
        probs = (-metric - low) / (high - low)
        np.clip(probs, 0., 1., out=probs)
        np.power(probs, prob_power, out=probs)

        neurons = np.flatnonzero(probs > 1e-3)
        cnt = 0
        # if apply_random_synaptogenesis:
        #     sampled_indices = neurons[self.rng.random(neurons.size) < probs[neurons]]
        #     self.try_grow_synapses_to_input(sampled_indices)
        #     cnt = sampled_indices.size
        if apply_random_synaptogenesis:
            self.rand_weights[neurons] = np.maximum(
                self.rand_weights[neurons], probs[neurons]
            )

        self.neurogenesis_queue[neurons] = np.maximum(
            self.neurogenesis_queue[neurons], probs[neurons]
        )
        return cnt

    def try_grow_synapses_to_input(self, neurons: SparseSdr):
        sampled_neurons = neurons[
            self.rng.random(neurons.size) < self.neurogenesis_queue[neurons]
        ]
        if sampled_neurons.size == 0:
            return

        self.rng.shuffle(sampled_neurons)

        found = None
        in_set = set(self.sparse_input)
        for neuron in sampled_neurons:
            to_choose_from = list(in_set - set(self.rf[neuron]))
            if to_choose_from:
                syn = self.rng.choice(to_choose_from)
                found = neuron, syn
                break

        if found is None:
            print('---SYN')
            return

        neuron, syn = found
        self.neurogenesis_queue[neuron] = 0.
        self.neurogenesis_cnt += 1

        to_change_ix = np.argmin(self.weights[neuron])
        self.assign_new_synapses(neuron, to_change_ix, syn)

    def assign_new_synapses(
            self, neurons: npt.NDArray[int] | int,
            to_change_ix: npt.NDArray[int] | int,
            new_synapses: npt.NDArray[int] | int
    ):
        raise NotImplementedError('Not implemented yet')

    def health_check(self):
        in_rate = self.fast_feedforward_trace.get()
        target_in_rate = in_rate.sum() / self.ff_size

        # relative to target input rate
        ip = in_rate / target_in_rate
        log_ip = np.log(ip)

        # relative to target input rate
        rfe_in = np.sum(ip[self.rf] * self.weights, axis=1)
        avg_rfe_in = rfe_in.mean()
        nrfe_in = rfe_in / avg_rfe_in
        log_nrfe_in = np.log(nrfe_in)

        out_rate = self.fast_output_trace.get()
        target_out_rate = out_rate.sum() / self.output_size

        # relative to target output rate
        op = out_rate / target_out_rate
        log_op = np.log(op)

        rfe_out = op / rfe_in
        avg_rfe_out = rfe_out.mean()
        nrfe_out = rfe_out / avg_rfe_out
        log_nrfe_out = np.log(nrfe_out)

        self.health_check_results = {
            'ln(ip)': log_ip,
            'ln(op)': log_op,
            'ln(nrfe_in)': log_nrfe_in,
            'avg(rfe_out)': avg_rfe_out,
            'ln(nrfe_out)': log_nrfe_out,
        }

        in_rate = self.slow_feedforward_trace.get()
        target_in_rate = in_rate.sum() / self.ff_size

        # relative to target input rate
        ip = in_rate / target_in_rate

        # relative to target input rate
        rfe_in = np.sum(ip[self.rf] * self.weights, axis=1)
        avg_rfe_in = rfe_in.mean()
        nrfe_in = rfe_in / avg_rfe_in
        log_nrfe_in = np.log(nrfe_in)

        self.slow_health_check_results = {
            'ln(ip)': log_ip,
            'ln(nrfe_in)': log_nrfe_in,
        }

    def get_health_check_stats(self, ff_trace, out_trace):
        # TODO:
        #   - Input's popularity IP — how specified input sdr represents the whole input
        #       in terms of popularity:
        #       ip[sdr] = in_rate[sdr].mean() / target_in_rate
        #   - Input Rate SDR popularity IRP — the same, but taking into account
        #       current Rate SDR rates:
        #       irp[sdr] = in_rate[sdr] * rate_sdr / target_in_rate
        #   - RF's efficiency RFE_i:
        #       rfe_i = in_rate[rf_i] * w_i
        #   - RFE^in_i:
        #       rfe^in_i = rfe_i / target_in_rate
        #   - Normalized RFE^in_i:
        #       nrfe^in_i = rfe^in_i / rfe^in.mean()
        #   - Learning Potential LP_i:
        #       lp_i = rfe^in_i / ip[rf_i] = rfe_i / in_rate[rf_i].mean()
        #   - Average IP:
        #       aip = ip[RF].mean()
        #   - Average LP:
        #       avg_lp = lp[RF].mean()
        #   - Output popularity OP_i:
        #       op_i = out_rate_i / target_out_rate
        #   - RFE^out_i:
        #       rfe^out_i = op_i / rfe^in_i
        #   - Normalized RFE^out_i:
        #       nrfe^out_i = rfe^out_i / rfe^out.mean()

        # TODO: check normalization — do I need it?
        in_rate = ff_trace.get()
        target_in_rate = in_rate.sum() / self.ff_size

        # relative to target input rate
        ip = in_rate / target_in_rate
        log_ip = np.log(ip)

        # relative to target input rate
        rfe_in = np.sum(ip[self.rf] * self.weights, axis=1)
        avg_rfe_in = rfe_in.mean()
        nrfe_in = rfe_in / avg_rfe_in
        log_nrfe_in = np.log(nrfe_in)

        rfp_in = np.mean(ip[self.rf], axis=1)
        avg_rfp_in = rfp_in.mean()
        log_rfp_in = np.log(rfp_in)

        lp = rfe_in / rfp_in
        avg_lp = lp.mean()
        log_lp = np.log(lp)

        out_rate = out_trace.get()
        target_out_rate = out_rate.sum() / self.output_size

        # relative to target output rate
        op = out_rate / target_out_rate
        log_op = np.log(op)

        rfe_out = op / rfe_in
        avg_rfe_out = rfe_out.mean()
        nrfe_out = rfe_out / avg_rfe_out
        log_nrfe_out = np.log(nrfe_out)
        appx_std_log_nrfe_out = np.mean(np.abs(log_nrfe_out))

        min_ln = np.log(1/10)
        return {
            'ln(ip)': np.maximum(log_ip, min_ln),
            'ln(op)': np.maximum(log_op, min_ln),

            'avg(rfe_in)': avg_rfe_in,
            'ln(nrfe_in)': np.maximum(log_nrfe_in, min_ln),

            'avg(rfp_in)': avg_rfp_in,
            'ln(rfp_in)': log_rfp_in,

            'avg(lp)': avg_lp,
            'ln(lp)': log_lp,

            'avg(rfe_out)': avg_rfe_out,
            'ln(nrfe_out)': np.maximum(log_nrfe_out, min_ln),
            'std(ln(nrfe_out))': appx_std_log_nrfe_out,
        }

    def on_end_newborn_phase(self):
        self.boosting_k = 0.
        self.neurogenesis_countdown = make_repeating_counter(self.prune_grow_schedule)
        print(f'Become adult: {self._state_str()}')

    def get_active_rf(self, weights):
        raise NotImplementedError('Not implemented yet')

    def get_target_rf_sparsity(self):
        raise NotImplementedError('Not implemented yet')

    @property
    def ff_size(self):
        raise NotImplementedError('Not implemented yet')

    @property
    def ff_avg_active_size(self):
        raise NotImplementedError('Not implemented yet')

    @property
    def ff_avg_sparsity(self):
        return self.ff_avg_active_size / self.ff_size

    @property
    def rf_size(self) -> int:
        raise NotImplementedError('Not implemented yet')

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
        return (
            # f'{self.rf_sparsity:.4f}'
            # f' | {self.rf_size}'
            f' | {self.learning_rate:.3f}'
            f' | {self.boosting_k:.2f}'
        )

    @property
    def feedforward_rate(self):
        raise NotImplementedError('Not implemented yet')

    @property
    def output_rate(self):
        return self.slow_output_trace.get()

    @property
    def output_relative_rate(self):
        target_rate = self.output_sds.sparsity
        return self.output_rate / target_rate

    @property
    def rf_match_trace(self):
        raise NotImplementedError('Not implemented yet')

    def output_entropy(self):
        return entropy(self.output_rate, sds=self.output_sds)

    @property
    def recognition_strength(self):
        return self.slow_recognition_strength_trace.get()

    def decay_stat_trackers(self):
        self.computation_speed.reset()
        self.slow_output_trace.reset()
        self.slow_recognition_strength_trace.reset()

        self.fast_output_trace.reset()


def normalize_weights(weights: npt.NDArray[float]):
    normalizer = np.abs(weights)
    if weights.ndim == 2:
        normalizer = normalizer.sum(axis=1, keepdims=True)
    else:
        normalizer = normalizer.sum()
    return np.clip(weights / normalizer, 0., 1)
