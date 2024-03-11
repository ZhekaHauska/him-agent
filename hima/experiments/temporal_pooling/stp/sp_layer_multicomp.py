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
from hima.experiments.temporal_pooling.stp.sp_layer import define_winners
from hima.experiments.temporal_pooling.stp.sp_utils import (
    RepeatingCountdown, make_repeating_counter
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

    # compartments
    compartments: dict
    compartments_ix: dict[str, int]
    compartments_weight: npt.NDArray[float]

    # potentiation and learning
    potentials: np.ndarray
    learning_rate: float
    synaptogenesis_countdown: RepeatingCountdown

    # output
    output_sds: Sds
    output_mode: OutputMode
    normalize_rates: bool

    sample_winners_frac: float
    winners: SparseSdr
    strongest_winner: int | None

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
            # learning
            learning_rate: float, learning_algo: str,
            synaptogenesis_cycle: float,
            # output
            output_sds: Sds, output_mode: str, normalize_rates: bool = True,
            sample_winners: float | None = None,
            slow_trace_decay: float = 0.99, fast_trace_decay: float = 0.95,
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

        self.output_sds = Sds.make(output_sds)
        self.output_mode = OutputMode[output_mode.upper()]
        self.normalize_rates = normalize_rates

        self.learning_algo = SpLearningAlgo[learning_algo.upper()]
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.modulation = 1.0

        self.sample_winners = sample_winners is not None and sample_winners > 0.
        self.sample_winners_frac = sample_winners if self.sample_winners else 0.

        self.winners = np.empty(0, dtype=int)
        self.winners_value = 1.0
        self.strongest_winner = None
        self.potentials = np.zeros(self.output_size)

        self.synaptogenesis_schedule = int(synaptogenesis_cycle / self.output_sds.sparsity)
        self.synaptogenesis_countdown = make_repeating_counter(self.synaptogenesis_schedule)
        self.synaptogenesis_cnt = 0
        self.synaptogenesis_score = np.zeros(self.output_size, dtype=float)

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
        self.accept_input(input_sdr)

        matched_input_activity = self.match_input(input_sdr, learn)
        self.select_winners()
        self.broadcast_winners()
        self.reinforce_winners(matched_input_activity, learn)

        output_sdr = self.select_output()
        self.accept_output(output_sdr, learn=learn)
        return output_sdr

    def accept_input(self, sdr: CompartmentsAnySparseSdr):
        """Accept new input and move to the next time step"""
        assert len(sdr) == len(self.compartments), f'{sdr=} | {self.compartments.keys()=}'

        # apply timed decay to neurons' potential
        self.potentials.fill(0.)

    def match_input(self, input_sdr: CompartmentsAnySparseSdr, learn):
        matched_input_activity = {}

        for comp_name in self.compartments:
            compartment = self.compartments[comp_name]
            compartment_weight = self.compartments_weight[self.compartments_ix[comp_name]]

            matched_input_activity[comp_name] = compartment.match_input(
                input_sdr[comp_name], learn=learn
            )
            # NB: compartment potentials contain only current time step induced potentials
            self.potentials += compartment_weight * compartment.potentials

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

    @property
    def output_size(self):
        return self.output_sds.size

    @property
    def is_newborn_phase(self):
        return False

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
