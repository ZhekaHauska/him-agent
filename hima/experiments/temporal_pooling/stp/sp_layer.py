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
from hima.common.utils import timed, safe_divide, isnone
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.sp import SpNewbornPruningMode
from hima.experiments.temporal_pooling.stp.sp_utils import (
    boosting, gather_rows,
    sample_for_each_neuron, RepeatingCountdown, tick, make_repeating_counter
)


class SpLearningAlgo(Enum):
    OLD = 1
    NEW = auto()
    NEW_SQ = auto()


class SpatialPooler:
    """A competitive network (as meant by Rolls)."""
    rng: Generator

    # I/O settings
    feedforward_sds: Sds
    adapt_to_ff_sparsity: bool

    output_sds: Sds
    output_mode: OutputMode

    initial_rf_sparsity: float
    target_max_rf_sparsity: float
    target_rf_to_input_ratio: float
    rf: npt.NDArray[int]
    weights: npt.NDArray[float]

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

    # cache
    sparse_input: SparseSdr
    dense_input: DenseSdr

    winners: SparseSdr
    strongest_winner: int | None
    potentials: np.ndarray

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
            prune_grow_cycle: float,
            # additional optional params
            normalize_rates: bool = True, sample_winners: bool = False,
            connectable_ff_size: int = None,
            # rand_weights_ratio: 0.7, rand_weights_noise: 0.01,
            slow_trace_decay: float = 0.95, fast_trace_decay: float = 0.75,
    ):
        self.rng = np.random.default_rng(seed)

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.adapt_to_ff_sparsity = adapt_to_ff_sparsity

        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode[output_mode.upper()]
        self.normalize_rates = normalize_rates

        self.learning_algo = SpLearningAlgo[learning_algo.upper()]
        self.stdp = self.get_learning_algos()[self.learning_algo]

        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate

        self.initial_rf_sparsity = min(
            initial_rf_to_input_ratio * self.feedforward_sds.sparsity,
            initial_max_rf_sparsity
        )
        self.target_rf_to_input_ratio = target_rf_to_input_ratio
        self.target_max_rf_sparsity = target_max_rf_sparsity

        rf_size = int(self.initial_rf_sparsity * self.ff_size)
        set_size = isnone(connectable_ff_size, self.ff_size)
        self.rf = sample_for_each_neuron(
            rng=self.rng, n_neurons=self.output_size,
            set_size=set_size, sample_size=rf_size
        )
        print(f'SP.layer init shape: {self.rf.shape} to {set_size}')
        self.weights = normalize_weights(
            self.rng.normal(loc=1.0, scale=0.0001, size=self.rf.shape)
        )
        # self.rand_weights = np.clip(
        #     self.rng.normal(loc=rand_weights_ratio, scale=0.01, size=self.output_size),
        #     0., 1.
        # )

        self.select_winners = self._sample_winners if sample_winners else self._select_winners

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

        self.sparse_input = []
        # use float not only to generalize to float-SDR, but also to eliminate
        # inevitable int-to-float converting when we multiply it by weights
        self.dense_input = np.zeros(self.ff_size, dtype=float)
        self.winners = []
        self.winners_value = 1.0
        self.strongest_winner = None
        self.potentials = np.zeros(self.output_size)

        # stats collection
        self.health_check_results = {}
        self.computation_speed = MeanValue(exp_decay=slow_trace_decay)
        # TODO: remove avg mass tracking
        self.slow_feedforward_trace = MeanValue(
            size=self.ff_size, track_avg_mass=True, exp_decay=slow_trace_decay
        )
        self.slow_feedforward_trace.put(self.feedforward_sds.sparsity)

        self.slow_feedforward_size_trace = MeanValue(exp_decay=slow_trace_decay)
        self.fast_feedforward_trace = MeanValue(
            size=self.ff_size, track_avg_mass=True, exp_decay=fast_trace_decay
        )
        self.fast_feedforward_trace.put(self.feedforward_sds.sparsity)

        self.slow_recognition_strength_trace = MeanValue(exp_decay=slow_trace_decay)
        self.slow_output_trace = MeanValue(
            size=self.output_size, track_avg_mass=True, exp_decay=slow_trace_decay
        )
        self.slow_output_trace.put(self.output_sds.sparsity)
        self.fast_output_trace = MeanValue(
            size=self.output_size, track_avg_mass=True, exp_decay=fast_trace_decay
        )
        self.fast_output_trace.put(self.output_sds.sparsity)

        self.neurogenesis_queue = np.zeros(self.output_size, dtype=int)

    def compute(self, input_sdr: AnySparseSdr, learn: bool = False) -> AnySparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.computation_speed.put(run_time)
        return output_sdr

    @timed
    def _compute(self, input_sdr: AnySparseSdr, learn: bool) -> AnySparseSdr:
        self.accept_input(input_sdr, learn=learn)
        self.try_activate_neurogenesis()

        matched_input_activity = self.match_current_input()
        delta_potentials = (matched_input_activity * self.weights).sum(axis=1)
        self.potentials += self.apply_boosting(delta_potentials)

        self.select_winners()
        self.reinforce_winners(matched_input_activity, learn)

        output_sdr = self.select_output()
        self.accept_output(output_sdr, learn=learn)

        return output_sdr

    def accept_input(self, sdr: AnySparseSdr, *, learn: bool):
        """Accept new input and move to the next time step"""
        sdr, value = split_sdr_values(sdr)

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

    def try_activate_neurogenesis(self):
        is_now, self.neurogenesis_countdown = tick(self.neurogenesis_countdown)
        if not is_now:
            return

        # self.health_check()
        if self.is_newborn_phase:
            self.shrink_receptive_field()
        else:
            self.prune_grow_synapses()
            pass

    def match_current_input(self, with_neurons: np.ndarray = None) -> npt.NDArray[float]:
        rf = self.rf if with_neurons is None else self.rf[with_neurons]
        return self.dense_input[rf]

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

        n_pot_winners = min(3*n_winners, self.potentials.size)
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

    def reinforce_winners(self, matched_input_activity, learn: bool):
        if not learn:
            return
        self.stdp(self.winners, matched_input_activity[self.winners])
        if not self.is_newborn_phase:
            self.try_grow_synapses_to_input(self.winners)

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
        post_rates = self.winners_value
        if self.output_mode == OutputMode.RATE:
            post_rates = np.expand_dims(self.winners_value, -1)

        lr = modulation * self.learning_rate

        w = self.weights[neurons]
        dw = lr * post_rates * (pre_rates - post_rates * w)

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
        post_rates = self.winners_value
        if self.output_mode == OutputMode.RATE:
            post_rates = np.expand_dims(self.winners_value, -1)

        lr = modulation * self.learning_rate

        w = self.weights[neurons]
        dw = lr * post_rates * (pre_rates - w)

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
        input_mass = self.dense_input[self.sparse_input].sum()
        self.slow_recognition_strength_trace.put(
            # safe_divide(self.potentials[sdr].mean(), input_mass)
            self.potentials[sdr].mean()
        )

        self.fast_output_trace.put(value, sdr)

    def process_feedback(self, feedback_sdr: SparseSdr, modulation: float = 1.0):
        # feedback SDR is the SP neurons that should be reinforced or punished
        fb_match_mask = self.match_current_input(with_neurons=feedback_sdr)
        # noinspection PyArgumentList
        self.stdp(feedback_sdr, fb_match_mask, modulation)

    def shrink_receptive_field(self):
        self.newborn_pruning_stage += 1

        if self.newborn_pruning_mode == SpNewbornPruningMode.LINEAR:
            new_sparsity = self.newborn_linear_progress(
                initial=self.initial_rf_sparsity, target=self.get_target_rf_sparsity()
            )
        elif self.newborn_pruning_mode == SpNewbornPruningMode.POWERLAW:
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
        self.weights = normalize_weights(
            gather_rows(self.weights, keep_connections_i)
        )
        self.learning_rate = self.newborn_linear_progress(
            initial=self.initial_learning_rate, target=0.2 * self.initial_learning_rate
        )
        self.boosting_k = self.newborn_linear_progress(
            initial=self.base_boosting_k, target=0.
        )
        print(f'Prune newborns: {self._state_str()}')

        self.decay_stat_trackers()

        if not self.is_newborn_phase:
            # it is ended
            self.on_end_newborn_phase()

        print(f'{self.output_entropy():.3f} | {self.recognition_strength:.1f}')

    def prune_grow_synapses(self):
        rate_threshold = 1 / 20

        avg_ff_trace = self.fast_feedforward_trace.get()
        ff_rate = avg_ff_trace / self.feedforward_sds.sparsity
        weighted_ff_rate = (self.weights * ff_rate[self.rf]).sum(axis=1)
        out_rate = self.fast_output_trace.get() / self.output_sds.sparsity

        pg_mask = weighted_ff_rate < rate_threshold
        non_pg_mask = ~pg_mask

        non_pg_nrp = out_rate[non_pg_mask] / weighted_ff_rate[non_pg_mask]
        avg_nrp = non_pg_nrp.mean()
        underperforming_neurons = non_pg_nrp / avg_nrp < rate_threshold

        to_pg_ix = np.flatnonzero(non_pg_mask)[underperforming_neurons]
        pg_mask[to_pg_ix] = True

        # what didn't change during finished cycle we change now,
        # and what we selected is for the change during the next cycle
        pg_ix = np.flatnonzero(self.neurogenesis_queue)
        self.neurogenesis_queue[pg_ix] = 0
        self.neurogenesis_queue[pg_mask] = 1

        to_change_ix = np.argmin(self.weights[pg_ix], axis=1)
        ff_sample_distr = safe_divide(avg_ff_trace, avg_ff_trace.sum())
        new_synapses = self.rng.choice(self.ff_size, size=len(pg_ix), p=ff_sample_distr)
        self.rf[pg_ix, to_change_ix] = new_synapses
        self.weights[pg_ix, to_change_ix] = 1.0 / self.rf_size
        self.weights[pg_ix] = normalize_weights(self.weights[pg_ix])

        print(
            f'{self.output_entropy():.3f}'
            f' | {self.recognition_strength:.1f}'
            f' | {avg_nrp:.3f}'
            f' | {np.count_nonzero(pg_mask)}'
        )

        self.decay_stat_trackers()

    def health_check(self):
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
        #       alp = lp[RF].mean()
        #   - Output popularity OP_i:
        #       op_i = out_rate_i / target_out_rate
        #   - RFE^out_i:
        #       rfe^out_i = op_i / rfe^in_i
        #   - Normalized RFE^out_i:
        #       nrfe^out_i = rfe^out_i / rfe^out.mean()

        # TODO: check normalization — do I need it?
        in_rate = self.fast_feedforward_trace.get()
        target_in_rate = in_rate.sum() / self.ff_size

        # relative to target input rate
        ip = in_rate / target_in_rate

        rfe = np.sum(in_rate[self.rf] * self.weights, axis=1)
        # relative to target input rate
        rfe_in = rfe / target_in_rate
        avg_rfe_in = rfe_in.mean()
        nrfe_in = rfe_in / avg_rfe_in

        ip_rf = np.mean(ip[self.rf], axis=1)
        lp = rfe_in / ip_rf
        alp = np.mean(lp)

        out_rate = self.fast_output_trace.get()
        target_out_rate = out_rate.sum() / self.output_size

        # relative to target output rate
        op = out_rate / target_out_rate
        rfe_out = op / rfe_in
        avg_rfe_out = rfe_out.mean()
        nrfe_out = rfe_out / avg_rfe_out

        slow_in_rate = self.slow_feedforward_trace.get()
        slow_target_in_rate = slow_in_rate.sum() / self.ff_size
        slow_rfe = np.sum(slow_in_rate[self.rf] * self.weights, axis=1)
        slow_rfe_in = slow_rfe / slow_target_in_rate

        self.health_check_results = {
            'ip': ip,
            'op': op,
            'avg_rfe_in': rfe_in.mean(),
            'nrfe_in': nrfe_in,
            'ip_rf': ip_rf,
            'avg_ip_rf': ip_rf.mean(),
            'lp': lp,
            'alp': alp,
            'avg_rfe_out': rfe_out.mean(),
            'nrfe_out': nrfe_out,
        }

        # I also care about:
        #   - rfe^in_i fast and slow (but the latter only for calcs);
        #       their thresholds low and high; how to map them to probs
        #   - rfe^out_i fast; their thresholds low and high; how to map them to probs;
        #       and std(rfe^out_i) or std for nrfe^out_i; how to map deviations to probs

    def try_grow_synapses_to_input(self, neurons: SparseSdr):
        if np.sum(self.neurogenesis_queue[neurons]) == 0:
            return

        found = False
        to_choose_from = np.flatnonzero(self.neurogenesis_queue[neurons])
        for _ in range(3):
            neuron = self.rng.choice(to_choose_from)
            syn = self.rng.choice(self.sparse_input)
            if syn not in self.rf[neuron]:
                found = True
                break

        if not found:
            return

        self.neurogenesis_queue[neuron] = 0

        to_change_ix = np.argmin(self.weights[neuron])
        self.rf[neuron, to_change_ix] = syn
        self.weights[neuron, to_change_ix] = 1.0 / self.rf_size
        self.weights[neuron] = normalize_weights(self.weights[neuron])

    def on_end_newborn_phase(self):
        self.boosting_k = 0.
        self.neurogenesis_countdown = make_repeating_counter(self.prune_grow_schedule)
        print(f'Become adult: {self._state_str()}')

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
        return round(self.slow_feedforward_size_trace.get())

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
        self.computation_speed.decay()
        self.slow_feedforward_trace.decay()
        self.slow_feedforward_size_trace.decay()
        self.slow_output_trace.decay()
        self.slow_recognition_strength_trace.decay()

        self.fast_feedforward_trace.decay()
        self.fast_output_trace.decay()

    def get_learning_algos(self):
        return {
            SpLearningAlgo.OLD: self._stdp,
            SpLearningAlgo.NEW: self._stdp_new,
            SpLearningAlgo.NEW_SQ: self._stdp_new_squared
        }


def normalize_weights(weights: npt.NDArray[float]):
    normalizer = np.abs(weights)
    if weights.ndim == 2:
        normalizer = normalizer.sum(axis=1, keepdims=True)
    else:
        normalizer = normalizer.sum()
    return np.clip(weights / normalizer, 0., 1)
