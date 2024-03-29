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

from hima.common.sdr import SparseSdr, DenseSdr
from hima.common.sdrr import RateSdr, AnySparseSdr, OutputMode, split_sdr_values
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.newborn_stage_controller import NewbornStageController
from hima.experiments.temporal_pooling.stp.sp_utils import (
    RepeatingCountdown, tick, make_repeating_counter, normalize_weights, define_winners
)


# Intermediate version of SP during the migration to work with both
# binary and rate encodings. Previous: sp_rate.py. Current last.
# New: supports synaptogenesis.

class SpLearningAlgo(Enum):
    OLD = 1
    NEW = auto()
    NEW_SQ = auto()


class SpatialPooler:
    """A competitive network (as meant by Rolls)."""
    rng: Generator

    # input
    feedforward_sds: Sds
    adapt_to_ff_sparsity: bool
    # input cache
    sparse_input: SparseSdr
    dense_input: DenseSdr

    # potentiation and learning
    potentials: np.ndarray
    learning_rate: float
    modulation: float

    # output
    output_sds: Sds
    output_mode: OutputMode

    winners: SparseSdr
    strongest_winner: int | None
    sample_winners: bool
    sample_winners_frac: float

    # connections
    initial_rf_sparsity: float
    target_max_rf_sparsity: float
    target_rf_to_input_ratio: float
    rf: npt.NDArray[int]
    weights: npt.NDArray[float]

    # newborn stage
    newborn_stage_controller: NewbornStageController

    # synaptogenesis
    synaptogenesis_schedule: int
    #   synaptogenesis score recalculation scheduler — defines the length of one cycle
    synaptogenesis_countdown: RepeatingCountdown
    #   tracks the number of synaptogenesis events in a cycle
    synaptogenesis_cnt: int
    #   synaptogenesis score is the willingness (~= probability) of a neuron to grow new synapses
    synaptogenesis_score: npt.NDArray[float]

    # stats
    #   average computation time
    computation_speed: MeanValue[float]
    #   input values accumulator
    slow_feedforward_trace: MeanValue[npt.NDArray[float]]
    #   input size accumulator
    slow_feedforward_size_trace: MeanValue[float]
    #   output values accumulator
    slow_output_trace: MeanValue[npt.NDArray[float]]
    #   recognition strength is an avg winners' overlap (potential)
    slow_recognition_strength_trace: MeanValue[float]

    def __init__(
            self, *, seed: int,
            feedforward_sds: Sds,
            adapt_to_ff_sparsity: bool,
            # initial — newborn; target — mature
            initial_max_rf_sparsity: float, target_max_rf_sparsity: float,
            initial_rf_to_input_ratio: float, target_rf_to_input_ratio: float,
            # output
            output_sds: Sds, output_mode: str,
            # learning
            learning_rate: float, learning_algo: str,
            # newborn phase
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            newborn_pruning_mode: str, boosting_k: float,
            # mature phase
            synaptogenesis_cycle: float,
            # additional optional params
            normalize_rates: bool = True, sample_winners: float | None = None,
            connectable_ff_size: int = False,
            # rand_weights_ratio: 0.7, rand_weights_noise: 0.01,
            slow_trace_decay: float = 0.99, fast_trace_decay: float = 0.95,
    ):
        self.rng = np.random.default_rng(seed)
        self.comp_name = None

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.adapt_to_ff_sparsity = adapt_to_ff_sparsity
        self.is_empty_input = True
        self.is_bound_input = False
        self.sparse_input = np.empty(0, dtype=int)
        # use float not only to generalize to Rate SDR, but also to eliminate
        # inevitable int-to-float converting when we multiply it by weights
        self.dense_input = np.zeros(self.ff_size, dtype=float)
        self.noise_input = 0.

        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode[output_mode.upper()]
        self.normalize_rates = normalize_rates
        self.activation_threshold = 0.

        self.potentials = np.zeros(self.output_size)
        self.learning_algo = SpLearningAlgo[learning_algo.upper()]
        self.learning_rate = learning_rate
        self.stdp = self.get_learning_algos()[self.learning_algo]
        self.modulation = 1.0

        self.winners = np.empty(0, dtype=int)
        self.winners_value = 1.0
        self.strongest_winner = None
        self.sample_winners = sample_winners is not None and sample_winners > 0.
        self.sample_winners_frac = sample_winners if self.sample_winners else 0.

        self.newborn_stage_controller = NewbornStageController(
            sp=self, newborn_pruning_cycle=newborn_pruning_cycle,
            newborn_pruning_stages=newborn_pruning_stages,
            newborn_pruning_mode=newborn_pruning_mode, boosting_k=boosting_k,
            initial_max_rf_sparsity=initial_max_rf_sparsity,
            target_max_rf_sparsity=target_max_rf_sparsity,
            initial_rf_to_input_ratio=initial_rf_to_input_ratio,
            target_rf_to_input_ratio=target_rf_to_input_ratio,
            connectable_ff_size=connectable_ff_size
        )
        # TODO: temporarily it is set by newborn stage controller, to be fixed
        # self.rf = sample_for_each_neuron(
        #     rng=self.rng, n_neurons=self.output_size,
        #     set_size=set_size, sample_size=rf_size
        # )

        self.weights = normalize_weights(
            self.rng.normal(loc=1.0, scale=0.0001, size=self.rf.shape)
        )
        self.rand_weights = np.zeros(self.output_size)

        self.synaptogenesis_schedule = int(synaptogenesis_cycle / self.output_sds.sparsity)
        # NB: synaptogenesis is active only during mature phase, thus initially
        # this scheduler represents newborn schedule, and then it is replaced
        # with mature synaptogenesis schedule (see on_end_newborn_phase)
        self.synaptogenesis_countdown = make_repeating_counter(
            self.newborn_stage_controller.newborn_pruning_schedule
        )
        self.synaptogenesis_score = np.zeros(self.output_size, dtype=float)
        self.synaptogenesis_cnt = 0

        # stats collection
        self.health_check_results = {}
        self.slow_health_check_results = {}

        self.computation_speed = MeanValue(exp_decay=slow_trace_decay)
        self.slow_feedforward_trace = MeanValue(
            size=self.ff_size, exp_decay=slow_trace_decay
        )
        self.slow_feedforward_trace.put(self.feedforward_sds.sparsity)

        self.slow_feedforward_size_trace = MeanValue(exp_decay=slow_trace_decay)
        self.fast_feedforward_trace = MeanValue(
            size=self.ff_size, exp_decay=fast_trace_decay
        )
        self.fast_feedforward_trace.put(self.feedforward_sds.sparsity)

        self.slow_recognition_strength_trace = MeanValue(exp_decay=slow_trace_decay)
        self.slow_output_trace = MeanValue(
            size=self.output_size, exp_decay=slow_trace_decay
        )
        self.slow_output_trace.put(self.output_sds.sparsity)
        self.fast_output_trace = MeanValue(
            size=self.output_size, exp_decay=fast_trace_decay
        )
        self.fast_output_trace.put(self.output_sds.sparsity)

        # NB: threshold trackers
        self.slow_output_size_trace = MeanValue(exp_decay=slow_trace_decay)
        self.slow_output_sdr_size_trace = MeanValue(exp_decay=slow_trace_decay)

        print(self.synaptogenesis_countdown)
        if not self.is_newborn_phase:
            self.on_end_newborn_phase()

    def compute(self, input_sdr: AnySparseSdr, learn: bool = False) -> AnySparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.computation_speed.put(run_time)
        return output_sdr

    @timed
    def _compute(self, input_sdr: AnySparseSdr, learn: bool) -> AnySparseSdr:
        self.accept_input(input_sdr, learn=learn)
        self.try_activate_synaptogenesis(learn)

        matched_input_activity = self.match_current_input()
        delta_potentials = (matched_input_activity * self.weights).sum(axis=1)
        self.apply_noisy_potentiation(delta_potentials)
        self.apply_boosting(delta_potentials)
        self.potentials += delta_potentials

        self.select_winners()
        self.reinforce_winners(matched_input_activity, learn)

        output_sdr = self.select_output()
        self.accept_output(output_sdr, learn=learn)

        return output_sdr

    def match_input(self):
        """Compute the output SDR."""
        matched_input_activity, run_time = self._match_input()
        self.computation_speed.put(run_time)
        return matched_input_activity

    @timed
    def _match_input(self):
        matched_input_activity = self.match_current_input()
        delta_potentials = (matched_input_activity * self.weights).sum(axis=1)

        # NB: synaptogenesis-induced noisy potentiation is a matter of each individual compartment!
        self.apply_noisy_potentiation(delta_potentials)

        # NB: apply newborn-phase boosting, which is a compartment-level effect
        # self.apply_boosting(delta_potentials)

        # NB2: potentials time-accumulation is a matter of neuron, not its compartments
        # thus, we always override the potentials each time step
        self.potentials[:] = delta_potentials

        return matched_input_activity

    def accept_input(self, sdr: AnySparseSdr, *, learn: bool):
        """Accept new input and move to the next time step"""
        sdr, value = split_sdr_values(sdr)
        self.is_empty_input = len(sdr) == 0

        # TODO: L2 norm
        l2_value = np.sqrt(np.sum(value**2))
        self.is_bound_input &= l2_value <= 1.0 + 1e-6

        # NB: in case input stream is not a Rate SDR, we should normalize it
        should_normalize = not self.is_bound_input and not np.isclose(l2_value, 0.)
        if not self.is_empty_input and should_normalize:
            value /= l2_value

        # NB: in case of Rate SDR, we induce a noise level — the rest of the input mass
        if self.is_bound_input:
            self.noise_input = 1. - l2_value
        else:
            self.noise_input = 0.

        # forget prev SDR
        self.dense_input[self.sparse_input] = 0.
        # apply timed decay to neurons' potential
        self.potentials.fill(0.)

        # set new SDR
        self.sparse_input = sdr
        self.dense_input[self.sparse_input] = value

        # For SP, an online learning is THE MOST natural operation mode.
        # We treat the opposite case as the special mode, which only partly affects SP state.
        if not learn:
            return

        self.slow_feedforward_trace.put(value, sdr)
        self.fast_feedforward_trace.put(value, sdr)

        self.slow_feedforward_size_trace.put(len(sdr))

    def get_step_debug_info(self):
        return {
            'potentials': np.sort(self.potentials),
            'recognition_strength': self.potentials[self.winners],
            'weights': self.weights,
            'rf': self.rf
        }

    def try_activate_synaptogenesis(self, learn):
        if not learn:
            return

        is_now, self.synaptogenesis_countdown = tick(self.synaptogenesis_countdown)
        print(self.synaptogenesis_countdown)
        if not is_now:
            return

        if self.is_newborn_phase:
            self.shrink_receptive_field()
        else:
            self.recalculate_synaptogenesis_score()

    def match_current_input(self, with_neurons: np.ndarray = None) -> npt.NDArray[float]:
        rf = self.rf if with_neurons is None else self.rf[with_neurons]
        return self.dense_input[rf]

    def apply_boosting(self, overlaps):
        self.newborn_stage_controller.apply_boosting(overlaps)

    def apply_noisy_potentiation(self, potentials):
        """
        Apply random noise to the neurons' potentials. Mutates the input array.

        It loosely approximates non-simulated weak connections. We treat neurons that
        are more efficient in average (hence, less synaptogenesis score) to have less
        noise mass, while in opposite case this mass is bigger.
        """
        if np.isclose(self.noise_input, 0.):
            return

        if self.noise_input < 0.2:
            return

        w_noise = self.noise_input / self.output_size**0.7
        potentials += self.rng.normal(loc=w_noise, scale=w_noise, size=self.output_size)

    def select_winners(self):
        if self.sample_winners:
            winners, strongest_winner = self._sample_winners()
        else:
            winners, strongest_winner = self._select_winners()

        self.winners, self.winners_value = define_winners(
            potentials=self.potentials, winners=winners, output_mode=self.output_mode,
            normalize_rates=self.normalize_rates, strongest_winner=self.strongest_winner
        )

    def _select_winners(self):
        n_winners = self.output_sds.active_size

        winners = np.argpartition(self.potentials, -n_winners)[-n_winners:]
        self.strongest_winner = cast(int, winners[np.argmax(self.potentials[winners])])
        winners.sort()
        return winners, self.strongest_winner

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
        return winners, self.strongest_winner

    def accept_winners(
            self, winners: SparseSdr,
            winners_value: float | npt.NDArray[float], strongest_winner: int
    ):
        self.winners = winners
        self.winners_value = winners_value
        self.strongest_winner = strongest_winner

    def reinforce_winners(self, matched_input_activity, learn: bool):
        if not learn or len(self.sparse_input) == 0:
            return

        # TODO: add LTD

        self.stdp(self.winners, matched_input_activity[self.winners])
        # if not self.is_newborn_phase:
        #     self.activate_synaptogenesis_step(self.winners)

    def _stdp(
            self, neurons: SparseSdr, pre_synaptic_activity: npt.NDArray[float],
            modulation: float = 1.0
    ):
        if len(neurons) == 0:
            return

        w = self.weights[neurons]
        n_matched = pre_synaptic_activity.sum(axis=1, keepdims=True) + .1
        lr = modulation * self.learning_rate / n_matched

        dw_matched = pre_synaptic_activity * lr

        self.weights[neurons] = normalize_weights(w + dw_matched)

    def _stdp_new(
            self, neurons: SparseSdr, pre_synaptic_activity: npt.NDArray[float],
            modulation: float = 1.0
    ):
        """
        Apply learning rule.

        Parameters
        ----------
        neurons: array of neurons affected with learning
        pre_synaptic_activity: dense array n_neurons x RF_size with their synaptic activations
        modulation: a modulation coefficient for the update step
        """
        if len(neurons) == 0:
            return

        pre_rates = pre_synaptic_activity
        # post_rates = self.winners_value
        post_rates = self.potentials[neurons]
        post_rates = np.expand_dims(post_rates, -1)

        lr = self.modulation * modulation * self.learning_rate

        w = self.weights[neurons]
        dw = lr * post_rates * (pre_rates - w)

        self.weights[neurons] = normalize_weights(w + dw)

    def _stdp_new_squared(
            self, neurons: SparseSdr, pre_synaptic_activity: npt.NDArray[float],
            modulation: float = 1.0
    ):
        """
        Apply learning rule.

        Parameters
        ----------
        neurons: array of neurons affected with learning
        pre_synaptic_activity: dense array n_neurons x RF_size with their synaptic activations
        modulation: a modulation coefficient for the update step
        """
        if len(neurons) == 0:
            return

        pre_rates = pre_synaptic_activity
        # post_rates = self.winners_value
        post_rates = self.potentials[neurons]
        post_rates = np.expand_dims(post_rates, -1)

        lr = modulation * self.learning_rate

        w = self.weights[neurons]
        dw = lr * post_rates * (pre_rates - post_rates * w)

        self.weights[neurons] = normalize_weights(w + dw)

    def select_output(self):
        if self.output_mode == OutputMode.RATE:
            return RateSdr(self.winners, values=self.winners_value)
        return self.winners

    def accept_output(self, sdr: SparseSdr, *, learn: bool):
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

        self.slow_output_sdr_size_trace.put(len(sdr))
        self.slow_output_size_trace.put(value.sum())

    def process_feedback(self, feedback_sdr: SparseSdr, modulation: float = 1.0):
        # feedback SDR is the SP neurons that should be reinforced or punished
        fb_match_mask = self.match_current_input(with_neurons=feedback_sdr)
        # noinspection PyArgumentList
        self.stdp(feedback_sdr, fb_match_mask, modulation)

    def shrink_receptive_field(self):
        self.newborn_stage_controller.shrink_receptive_field()
        self.decay_stat_trackers()

        if not self.is_newborn_phase:
            # it is ended
            self.on_end_newborn_phase()

        print(f'{self.output_entropy():.3f} | {self.recognition_strength:.1f}')

    def recalculate_synaptogenesis_score(self):
        # NB: usually we work in log-space => log_ prefix is mostly omit for vars
        self.health_check()

        self.synaptogenesis_score[:] = 0.
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
            f' | {self.synaptogenesis_cnt}'
            f' | {rnd_cnt}'
            f' | {np.sum(self.synaptogenesis_score):.2f}'
        )

        self.synaptogenesis_cnt = 0
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
        if apply_random_synaptogenesis:
            self.rand_weights[neurons] = np.maximum(
                self.rand_weights[neurons], probs[neurons]
            )

        self.synaptogenesis_score[neurons] = np.maximum(
            self.synaptogenesis_score[neurons], probs[neurons]
        )
        return cnt

    def activate_synaptogenesis_step(self, neuron):
        in_set = set(self.sparse_input)
        to_choose_from = list(in_set - set(self.rf[neuron]))
        if not to_choose_from:
            return False

        syn = self.rng.choice(to_choose_from)
        to_change_ix = np.argmin(self.weights[neuron])
        self.assign_new_synapses(neuron, to_change_ix, syn)
        return True

    def resample_synapses(self, neurons: npt.NDArray[int]):
        ip = np.exp(self.slow_health_check_results['ln(ip)'])
        ff_sample_distr = ip / ip.sum()
        new_synapses = self.rng.choice(self.ff_size, size=neurons.size, p=ff_sample_distr)
        to_change_ix = np.argmin(self.weights[neurons], axis=1)

        self.assign_new_synapses(neurons, to_change_ix, new_synapses)

    def assign_new_synapses(
            self, neurons: npt.NDArray[int] | int,
            to_change_ix: npt.NDArray[int] | int,
            new_synapses: npt.NDArray[int] | int
    ):
        self.rf[neurons, to_change_ix] = new_synapses
        self.weights[neurons, to_change_ix] = 1.0 / self.rf_size
        self.weights[neurons] = normalize_weights(self.weights[neurons])

    def health_check(self):
        # current Input Rate
        in_rate = self.fast_feedforward_trace.get()
        # Target Input Rate: average rate of each presynaptic neuron
        target_in_rate = in_rate.sum() / self.ff_size

        # NB: Most of the following metrics are relative to some target metric
        # NB2: log-space is used to make the metrics more linear (easier plots and separation)

        # IP (Input Popularity): the relative frequency of each presynaptic neuron
        ip = in_rate / target_in_rate
        log_ip = np.log(ip)

        # RFE^in (Receptive Field Efficiency for matching input):
        # how well each neuron's RF tuned to the input's distribution
        rfe_in = np.sum(ip[self.rf] * self.weights, axis=1)
        avg_rfe_in = rfe_in.mean()
        # NRFE^in (Normalized RFE^in): RFE^in relative to its average
        nrfe_in = rfe_in / avg_rfe_in
        log_nrfe_in = np.log(nrfe_in)

        # current Output Rate
        out_rate = self.fast_output_trace.get()
        # Target Output Rate: average rate of each postsynaptic neuron
        # NB: for binary output = sparsity, but for rate output this does not hold
        target_out_rate = out_rate.sum() / self.output_size

        # OP (Output Popularity): the relative frequency of each postsynaptic neuron
        op = out_rate / target_out_rate
        log_op = np.log(op)

        # RFE^out (Receptive Field Efficiency for activating neuron):
        # how well each neuron's RF tuning translates to the neuron's activation
        rfe_out = op / rfe_in
        avg_rfe_out = rfe_out.mean()
        # NRFE^out (Normalized RFE^out): RFE^out relative to its average
        nrfe_out = rfe_out / avg_rfe_out
        log_nrfe_out = np.log(nrfe_out)

        self.health_check_results = {
            'ln(ip)': log_ip,
            'ln(op)': log_op,
            'ln(nrfe_in)': log_nrfe_in,
            'avg(rfe_out)': avg_rfe_out,
            'ln(nrfe_out)': log_nrfe_out,
        }

        # NB: for synaptogenesis it is better to track both fast and slow pacing stats

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
        # NB: the following metrics are from the health_check method + some additional
        # helpful metrics for analysis

        in_rate = ff_trace.get()
        target_in_rate = in_rate.sum() / self.ff_size

        ip = in_rate / target_in_rate
        log_ip = np.log(ip)

        rfe_in = np.sum(ip[self.rf] * self.weights, axis=1)
        avg_rfe_in = rfe_in.mean()
        nrfe_in = rfe_in / avg_rfe_in
        log_nrfe_in = np.log(nrfe_in)

        # RFP^in (Receptive Field Popularity):
        # NB: compared to RFE^in, it uses the "uniformly distributed weights" instead of
        # real weights.
        # NB2: alone, avg RFP^in shows how much neurons connect to the strong (=frequent) input.
        # Ideally, this value should NOT be very high, because this means that the neurons
        # are not selective enough, and just connect to the most frequent input.
        rfp_in = np.mean(ip[self.rf], axis=1)
        avg_rfp_in = rfp_in.mean()
        log_rfp_in = np.log(rfp_in)

        # LP (Learning Potential):
        # how well each neuron's RF weights are tuned relative to "un-tuned" uniform weights
        # NB: if below 1, then the neuron's learning performs worse than no learning.
        lp = rfe_in / rfp_in
        avg_lp = lp.mean()
        log_lp = np.log(lp)

        out_rate = out_trace.get()
        target_out_rate = out_rate.sum() / self.output_size

        op = out_rate / target_out_rate
        log_op = np.log(op)

        rfe_out = op / rfe_in
        avg_rfe_out = rfe_out.mean()
        nrfe_out = rfe_out / avg_rfe_out
        log_nrfe_out = np.log(nrfe_out)
        # NB: very rough approximate of the STD[log(nrfe_out)] — enough to see dynamics and scale
        appx_std_log_nrfe_out = np.mean(np.abs(log_nrfe_out))

        # for plotting, clip values to avoid uninformative plot scales
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
        self.synaptogenesis_countdown = make_repeating_counter(self.synaptogenesis_schedule)
        print(f'Become adult: {self.sng_state_str()}')

    def get_active_rf(self, weights):
        w_thr = 1 / self.rf_size
        return weights >= w_thr

    def get_target_rf_sparsity(self):
        return self.newborn_stage_controller.get_target_rf_sparsity()

    @property
    def ff_size(self):
        return self.feedforward_sds.size

    @property
    def ff_avg_active_size(self):
        return round(self.slow_feedforward_size_trace.get())

    @property
    def ff_avg_sparsity(self):
        return self.ff_avg_active_size / self.ff_size

    @property
    def out_avg_active_size(self):
        return round(self.slow_output_size_trace.get())

    @property
    def out_avg_sparsity(self):
        return self.out_avg_active_size / self.output_size

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
        return self.newborn_stage_controller.is_newborn_phase

    def sng_state_str(self) -> str:
        return (
            f'{self.rf_sparsity:.4f}'
            f' | {self.rf_size}'
            f' | {self.learning_rate:.3f}'
            f' | {self.newborn_stage_controller.boosting_k:.2f}'
            f' | {self.synaptogenesis_countdown[0]}'
        )

    @property
    def feedforward_rate(self):
        return self.slow_feedforward_trace.get()

    @property
    def output_rate(self):
        return self.slow_output_trace.get()

    @property
    def output_relative_rate(self):
        target_rate = self.output_sds.sparsity
        return self.output_rate / target_rate

    @property
    def rf_match_trace(self):
        return self.slow_feedforward_trace.get()[self.rf]

    def output_entropy(self):
        return entropy(self.output_rate, sds=self.output_sds)

    @property
    def recognition_strength(self):
        return self.slow_recognition_strength_trace.get()

    def decay_stat_trackers(self):
        self.computation_speed.reset()
        self.slow_feedforward_trace.reset()
        self.slow_feedforward_size_trace.reset()
        self.slow_output_trace.reset()
        self.slow_recognition_strength_trace.reset()

        self.fast_feedforward_trace.reset()
        self.fast_output_trace.reset()

    def get_learning_algos(self):
        return {
            SpLearningAlgo.OLD: self._stdp,
            SpLearningAlgo.NEW: self._stdp_new,
            SpLearningAlgo.NEW_SQ: self._stdp_new_squared
        }
