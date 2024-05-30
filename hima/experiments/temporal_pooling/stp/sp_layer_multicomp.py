#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

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
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import entropy
from hima.experiments.temporal_pooling.stp.sp_utils import (
    RepeatingCountdown, make_repeating_counter, tick
)


# Multi-compartmental version of Spatial Encoder. Works on top of the sp_layer compartments.


# TODO:
#   - [ ] SdrCache (dense+sparse datastruct)


class SpatialPooler:
    """A competitive network (as meant by Rolls)."""
    rng: Generator

    # compartments
    compartments: dict
    compartments_ix: dict[str, int]
    compartments_weight: npt.NDArray[float]
    product_weight: float

    # potentiation and learning
    potentials: npt.NDArray[float]
    learning_rate: float
    modulation: float
    synaptogenesis_stats_update_countdown: RepeatingCountdown

    # output
    output_sds: Sds
    output_mode: OutputMode

    activation_threshold: float
    activation_threshold_learn_countdown: RepeatingCountdown
    winners: SparseSdr
    winners_value: float | npt.NDArray[float]

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
            product_weight: float,
            # learning
            learning_rate: float,
            synaptogenesis_cycle: float,
            # output
            output_sds: Sds, output_mode: str,
    ):
        self.rng = np.random.default_rng(seed)
        print(f'{output_sds=}')

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
            self.compartments[comp_name].comp_name = comp_name
        self.product_weight = product_weight

        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode[output_mode.upper()]

        self.potentials = np.zeros(self.output_size)
        # aux cache for multiplicative potentials
        self.prod_potentials = np.ones(self.output_size)
        self.learning_rate = learning_rate
        self.modulation = 1.0

        self.raw_logits = np.zeros(self.output_size)
        self.activation_threshold = 0.0
        self._predict_mode = False
        self.winners = np.empty(0, dtype=int)
        self.winners_value = 1.0

        self.synaptogenesis_stats_update_schedule = int(
            synaptogenesis_cycle / self.output_sds.sparsity
        )
        self.synaptogenesis_stats_update_countdown = make_repeating_counter(
            self.synaptogenesis_stats_update_schedule
        )
        self.synaptogenesis_event_prob = np.clip(
            5.0 / synaptogenesis_cycle, 0.0, 1.0
        )
        self.synaptogenesis_score = np.zeros(self.output_size, dtype=float)
        self.synaptogenesis_cnt = 0
        self.synaptogenesis_cnt_successful = 0

        # stats collection
        self.health_check_results = {}
        self.slow_health_check_results = {}

        slow_lr = LearningRateParam(window=400_000)
        fast_lr = LearningRateParam(window=40_000)
        very_fast_lr = LearningRateParam(window=400)

        self.computation_speed = MeanValue(lr=slow_lr)
        self.slow_recognition_strength_trace = MeanValue(lr=slow_lr)
        self.slow_output_trace = MeanValue(
            size=self.output_size, lr=slow_lr, initial_value=self.output_sds.sparsity
        )
        self.fast_output_trace = MeanValue(
            size=self.output_size, lr=fast_lr, initial_value=self.output_sds.sparsity
        )

        # NB: threshold trackers
        self.slow_output_size_trace = MeanValue(lr=slow_lr)
        self.fast_output_sdr_size_trace = MeanValue(
            lr=very_fast_lr, initial_value=self.output_sds.active_size
        )
        self.activation_threshold_learn_countdown = make_repeating_counter(
            self.fast_output_sdr_size_trace.safe_window
        )

        # synchronize selected attributes for compartments with the main SP
        for comp_name in self.compartments:
            self.compartments[comp_name].output_mode = self.output_mode

    def compute(self, input_sdr: CompartmentsAnySparseSdr, learn: bool = False) -> AnySparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._compute(input_sdr, learn)
        self.computation_speed.put(run_time)
        return output_sdr

    def predict(self, input_sdr: CompartmentsAnySparseSdr) -> AnySparseSdr:
        """Compute the output SDR."""
        output_sdr, run_time = self._predict(input_sdr)
        self.computation_speed.put(run_time)
        return output_sdr

    @timed
    def _compute(self, input_sdr: CompartmentsAnySparseSdr, learn: bool) -> AnySparseSdr:
        self._predict_mode = False
        self.accept_input(input_sdr, learn=learn)

        matched_input_activity = self.match_input()
        self.select_winners()
        self.broadcast_winners()
        self.reinforce_winners(matched_input_activity, learn)

        output_sdr = self.select_output()
        self.accept_output(output_sdr, learn=learn)

        self.try_activate_synaptogenesis(learn)

        return output_sdr

    @timed
    def _predict(self, input_sdr: CompartmentsAnySparseSdr) -> AnySparseSdr:
        self._predict_mode = True
        self.accept_input(input_sdr, learn=False)

        self.match_input()
        self.select_winners()
        self.broadcast_winners()

        output_sdr = self.select_output()
        self.accept_output(output_sdr, learn=False)
        return output_sdr

    def accept_input(self, sdr: CompartmentsAnySparseSdr, *, learn: bool):
        """Accept new input and move to the next time step"""
        assert len(sdr) == len(self.compartments), f'{sdr=} | {self.compartments.keys()=}'
        for comp_name in self.compartments:
            self.compartments[comp_name].accept_input(sdr[comp_name], learn=learn)

        # apply timed decay to neurons' potential
        self.potentials.fill(0.)
        # ff_sdr, _ = split_sdr_values(sdr['feedforward'])
        # if len(ff_sdr) > 0:
        #     self.potentials *= 0.5
        # else:
        #     self.potentials.fill(0.)

    def match_input(self):
        matched_input_activity = {}

        self.prod_potentials.fill(self.product_weight)
        for comp_name in self.compartments:
            compartment = self.compartments[comp_name]
            compartment_weight = self.compartments_weight[self.compartments_ix[comp_name]]

            matched_input_activity[comp_name] = compartment.match_input()
            # NB: compartment potentials contain only current time step induced potentials
            self.potentials += compartment_weight * compartment.potentials
            self.prod_potentials *= compartment.potentials

        self.potentials *= 1 - self.product_weight
        self.potentials += self.prod_potentials

        return matched_input_activity

    def broadcast_winners(self):
        for comp_name in self.compartments:
            self.compartments[comp_name].accept_winners(
                winners=RateSdr(self.winners, self.winners_value)
            )

    def reinforce_winners(self, matched_input_activity, learn: bool):
        if not learn:
            return

        for comp_name in self.compartments:
            self.compartments[comp_name].reinforce_winners(
                matched_input_activity[comp_name], learn
            )

    # noinspection PyMethodMayBeStatic
    def get_step_debug_info(self):
        # its existence is required for trackers
        return {}

    def try_activate_synaptogenesis(self, learn):
        if not learn:
            return

        self.try_activate_synaptogenesis_step()

        is_now, self.synaptogenesis_stats_update_countdown = tick(
            self.synaptogenesis_stats_update_countdown
        )
        if not is_now:
            return

        are_all_mature = not any(
            compartment.is_newborn_phase
            for compartment in self.compartments.values()
        )
        if are_all_mature:
            self.recalculate_synaptogenesis_score()

    def select_winners(self):
        self.winners, self.winners_value = self._select_winners_by_threshold()

    def _select_winners_by_threshold(self):
        logits = self.potentials ** 2
        l2_logits = logits.sum()
        if not np.isclose(l2_logits, 0.):
            logits /= np.sqrt(l2_logits)

        # logits = softmax(self.potentials)

        if self._predict_mode:
            threshold = 0.01
        else:
            threshold = self.activation_threshold

        winners = np.flatnonzero(logits >= threshold)
        winners_value = logits[winners]

        # NB: output normalization???
        # if len(winners) > 0:
        #     winners_value /= winners_value.sum()

        return winners, winners_value

    def select_output(self):
        if self.output_mode == OutputMode.RATE:
            return RateSdr(self.winners, values=self.winners_value)
        return self.winners

    def accept_output(self, sdr: SparseSdr, *, learn: bool):
        for comp_name in self.compartments:
            self.compartments[comp_name].accept_output(sdr, learn=learn)

        sdr, value = split_sdr_values(sdr)
        if not learn:
            return

        # update winners activation stats
        self.slow_output_trace.put(value, sdr)

        # FIXME: make two metrics: for pre-weighting, post weighting delta
        avg_winner_potential = self.potentials[sdr].mean() if len(sdr) > 0 else 0.
        self.slow_recognition_strength_trace.put(avg_winner_potential)

        self.fast_output_trace.put(value, sdr)

        self.fast_output_sdr_size_trace.put(len(sdr))
        self.slow_output_size_trace.put(value.sum())

        is_now, self.activation_threshold_learn_countdown = tick(
            self.activation_threshold_learn_countdown
        )
        if is_now:
            n_winners = self.fast_output_sdr_size_trace.get()
            # print(f'{len(self.winners)} | {n_winners:.1f} | {self.activation_threshold:.5f}')
            err = np.clip(n_winners - self.output_sds.active_size, -5.0,5.0)
            self.activation_threshold += err * 0.0001

    def process_feedback(self, feedback_sdr: SparseSdr, modulation: float = 1.0):
        raise NotImplementedError('Not implemented yet')

    def try_activate_synaptogenesis_step(self):
        avg_winner_recognition_str = self.slow_recognition_strength_trace.get()
        if len(self.winners) > 0:
            winner_recognition_str = max(self.potentials[self.winners].mean(), 1e-6)
        else:
            winner_recognition_str = avg_winner_recognition_str

        beta = avg_winner_recognition_str / winner_recognition_str
        sng_prob = min(1.0, beta * self.synaptogenesis_event_prob)

        if self.rng.random() >= sng_prob:
            return

        # select the winner for the synaptogenesis
        ix_enabled = np.flatnonzero(self.synaptogenesis_score)
        if len(ix_enabled) == 0:
            return

        noisy_potentials = np.abs(self.rng.normal(
            loc=self.potentials[ix_enabled],
            scale=self.synaptogenesis_score[ix_enabled]
        ))
        sng_winner = ix_enabled[np.argmax(noisy_potentials)]

        # select the compartment for the synaptogenesis
        nrfe_in_compartment = self.health_check_results['nrfe_in_cmp'][:, sng_winner]
        compartment_probs = 1 / nrfe_in_compartment
        compartment_probs /= compartment_probs.sum()
        compartments = list(self.compartments.values())
        compartment = self.rng.choice(compartments, p=compartment_probs)

        success = compartment.activate_synaptogenesis_step(sng_winner)
        self.synaptogenesis_score[sng_winner] = 0.
        self.synaptogenesis_cnt += 1
        self.synaptogenesis_cnt_successful += int(success)

    def recalculate_synaptogenesis_score(self):
        # NB: usually we work in log-space => log_ prefix is mostly omit for vars
        self.health_check()

        self.synaptogenesis_score[:] = 0.

        # TODO: optimize thresholds and power scaling;
        #   also remember cumulative effect of prob during the cycle for frequent neurons
        abs_rate_low, abs_rate_high = np.log(20), np.log(100)
        eff_low, eff_high = np.log(1.5), np.log(20)

        # Step 1: deal with absolute rates: sleepy input RF and output
        # TODO: consider increasing random potential part for them
        self.apply_synaptogenesis_to_metric(
            self.slow_health_check_results['ln(nrfe_in)'],
            abs_rate_low, abs_rate_high,
            prob_power=2.0,
        )

        # Step 2: deal with sleepy output
        self.apply_synaptogenesis_to_metric(
            self.health_check_results['ln(nrfe_out)'],
            eff_low, eff_high,
            prob_power=1.5,
        )

        print(
            f'+ {self.output_entropy():.3f}'
            f' | {self.recognition_strength:.1f}'
            f' | {self.health_check_results["avg(rfe_out)"]:.3f}'
            f' || {self.synaptogenesis_cnt}'
            f' | {safe_divide(self.synaptogenesis_cnt_successful, self.synaptogenesis_cnt):.2f}'
            f' | {np.sum(self.synaptogenesis_score):.2f}'
            f' || {self.activation_threshold:.2f}'
            f' | {self.fast_output_sdr_size_trace.get():.1f}'
            f' | {self.slow_output_size_trace.get():.2f}'
        )

        self.synaptogenesis_cnt = 0
        self.synaptogenesis_cnt_successful = 0

    def apply_synaptogenesis_to_metric(
            self, metric: npt.NDArray[float], low: float, high: float, prob_power: float = 1.5,
    ):
        probs = (-metric - low) / (high - low)
        np.clip(probs, 0., 1., out=probs)
        np.power(probs, prob_power, out=probs)

        neurons = np.flatnonzero(probs > 1e-3)

        self.synaptogenesis_score[neurons] = np.maximum(
            self.synaptogenesis_score[neurons], probs[neurons]
        )

    def health_check(self):
        nrfe_in_compartment = []
        for compartment in self.compartments.values():
            # Part 1: fast metrics
            in_rate = compartment.fast_feedforward_trace.get()
            target_in_rate = in_rate.sum() / compartment.ff_size

            # relative to target input rate
            ip = in_rate / target_in_rate

            # relative to target input rate
            rfe_in = np.sum(ip[compartment.rf] * compartment.weights, axis=1)
            avg_rfe_in = rfe_in.mean()
            nrfe_in = rfe_in / avg_rfe_in

            nrfe_in_compartment.append(nrfe_in)

        nrfe_in_compartment = np.stack(nrfe_in_compartment)
        avg_nrfe_in = np.prod(nrfe_in_compartment, axis=0)
        log_avg_nrfe_in = np.log(avg_nrfe_in)

        out_rate = self.fast_output_trace.get()
        target_out_rate = out_rate.sum() / self.output_size

        # relative to target output rate
        op = out_rate / target_out_rate
        log_op = np.log(op)

        rfe_out = op / avg_nrfe_in
        avg_rfe_out = rfe_out.mean()
        nrfe_out = rfe_out / avg_rfe_out
        log_nrfe_out = np.log(nrfe_out)

        self.health_check_results = {
            'nrfe_in_cmp': nrfe_in_compartment,
            'ln(nrfe_in)': log_avg_nrfe_in,
            'ln(op)': log_op,
            'avg(rfe_out)': avg_rfe_out,
            'ln(nrfe_out)': log_nrfe_out,
        }

        nrfe_in_compartment = []
        for compartment in self.compartments.values():
            # Part 2: slow metrics
            in_rate = compartment.slow_feedforward_trace.get()
            target_in_rate = in_rate.sum() / compartment.ff_size

            # relative to target input rate
            ip = in_rate / target_in_rate

            # relative to target input rate
            rfe_in = np.sum(ip[compartment.rf] * compartment.weights, axis=1)
            avg_rfe_in = rfe_in.mean()
            nrfe_in = rfe_in / avg_rfe_in

            nrfe_in_compartment.append(nrfe_in)

        avg_nrfe_in = np.prod(np.stack(nrfe_in_compartment), axis=0)
        log_avg_nrfe_in = np.log(avg_nrfe_in)

        self.slow_health_check_results = {
            # 'ln(ip)': log_ip,
            'ln(nrfe_in)': log_avg_nrfe_in,
        }

    def get_health_check_stats(self, ff_trace, out_trace):
        raise NotImplementedError('Not implemented yet')

    @property
    def output_size(self):
        return self.output_sds.size

    @property
    def is_newborn_phase(self):
        return False

    @property
    def output_rate(self):
        return self.slow_output_trace.get()

    @property
    def output_relative_rate(self):
        target_rate = self.output_sds.sparsity
        return self.output_rate / target_rate

    def output_entropy(self):
        return entropy(self.output_rate)

    @property
    def recognition_strength(self):
        return self.slow_recognition_strength_trace.get()

    def decay_stat_trackers(self):
        self.computation_speed.reset()
        self.slow_output_trace.reset()
        self.slow_recognition_strength_trace.reset()

        self.fast_output_trace.reset()
