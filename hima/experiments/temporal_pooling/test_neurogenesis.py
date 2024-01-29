#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.run.wandb import get_logger
from hima.common.sds import TSdsShortNotation, Sds
from hima.common.timer import timer, print_with_timestamp
from hima.common.utils import isnone, prepend_dict_keys
from hima.experiments.temporal_pooling.data.synthetic_patterns import (
    sample_sdr,
    sample_noisy_sdr, sample_rate_sdr, sample_noisy_sdr_rate_sdr, sample_noisy_rates_rate_sdr
)
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.stats.metrics import (
    sequence_similarity_elementwise,
    sdr_similarity
)
from hima.experiments.temporal_pooling.stats.sdr_tracker import SdrTracker
from hima.experiments.temporal_pooling.stats.sp_tracker import SpTracker
from hima.experiments.temporal_pooling.utils import resolve_random_seed

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class NeurogenesisExperiment:
    config: GlobalConfig
    logger: Run | None
    init_time: float

    seed: int

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int,
            binary: bool,
            input_sds: TSdsShortNotation,
            output_sds: TSdsShortNotation,
            n_prototypes: int,
            visible_frac: float,
            noise_level: float,
            n_epochs: int,
            n_steps: int,

            n_seq_elements: int,
            seq_logging_schedule: int,
            n_sim_elements: int,
            sim_noise_level: list[int],

            step_flush_schedule: int,
            aggregate_flush_schedule: int,
            sp_potentials_quantile: float,

            rates_temp: float,

            layer: TConfig,

            # data: TConfig,
            project: str = None, wandb_init: TConfig = None,
            **_
    ):
        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=StpLazyTypeResolver()
        )
        self.logger = self.config.resolve_object(
            isnone(wandb_init, {}),
            object_type_or_factory=get_logger,
            config=config, log=log, project=project
        )
        self.seed = resolve_random_seed(seed)
        self.rng = np.random.default_rng(self.seed)

        self.input_sds = Sds.make(input_sds)
        self.output_sds = Sds.make(output_sds)

        self.binary = binary
        self.n_prototypes = n_prototypes
        self.noise_level = noise_level
        self.n_epochs = n_epochs
        self.n_steps = n_steps

        self.n_seq_elements = n_seq_elements
        self.seq_logging_schedule = seq_logging_schedule
        self.n_sim_elements = n_sim_elements
        self.sim_noise_level = sim_noise_level

        self.input_dense_cache = np.zeros(self.input_sds.size, dtype=float)
        self.output_dense_cache = np.zeros(self.output_sds.size, dtype=float)

        self.data = [
            sample_sdr(self.rng, self.input_sds)
            for _ in range(2 * self.n_prototypes)
        ]
        if not self.binary:
            self.data = [
                sample_rate_sdr(self.rng, sdr, rates_temp)
                for sdr in self.data
            ]

        # list of (starting index in data, number of examples) for each experiment stage
        self.data_parts = [
            (0, self.n_prototypes),
            (self.n_prototypes, self.n_prototypes),
            (0, 2 * self.n_prototypes)
        ]

        # list of (start, end) indices in input_sds for each experiment stage
        shift = round(visible_frac * self.input_sds.size)
        self.visible_sds_ranges = [
            (0, shift),
            (self.input_sds.size - shift, self.input_sds.size),
            None
        ]

        self.epoch = 0
        self.metrics = dict()

        self.layer = self.config.resolve_object(
            layer, feedforward_sds=self.input_sds, output_sds=self.output_sds,
        )

        if not self.logger:
            return

        sdr_tracker_config = dict(
            step_flush_schedule=step_flush_schedule,
            aggregate_flush_schedule=aggregate_flush_schedule
        )
        self.input_sdr_tracker = SdrTracker(self.input_sds, **sdr_tracker_config)
        self.output_sdr_tracker = SdrTracker(self.output_sds, **sdr_tracker_config)
        self.sp_tracker = SpTracker(
            self.layer, step_flush_schedule=step_flush_schedule,
            potentials_quantile=sp_potentials_quantile
        )

        self.sim_test_sequences = self.generate_sim_test_sequences()
        self.input_mx = self.get_similarity_matrix(self.sim_test_sequences, self.input_dense_cache)

        if self.binary:
            # SDR
            self.sim_test_sdrs = [
                [
                    (sdr, sample_noisy_sdr(self.rng, self.input_sds, sdr, noise_level))
                    for sdr in self.rng.choice(self.data, size=n_sim_elements, replace=False)
                ]
                for noise_level in sim_noise_level
            ]
        else:
            # Rate SDR
            self.sim_test_sdrs = []
            for noise_level in sim_noise_level:
                rate_sdrs = self.rng.choice(self.data, size=n_sim_elements, replace=False)
                seq = []
                for rate_sdr in rate_sdrs:
                    if self.rng.random() < 0.5:
                        # noisy sdr
                        noisy_rate_sdr = sample_noisy_sdr_rate_sdr(
                            self.rng, self.input_sds, rate_sdr, noise_level
                        )
                    else:
                        # noisy rate
                        noisy_rate_sdr = sample_noisy_rates_rate_sdr(
                            self.rng, rate_sdr, noise_level
                        )
                    seq.append((rate_sdr, noisy_rate_sdr))

                self.sim_test_sdrs.append(seq)

        self.input_avg_sims = [
            np.mean([
                sdr_similarity(sdr, noisy_sdr, symmetrical=True, dense_cache=self.input_dense_cache)
                for sdr, noisy_sdr in sim_test_sdr_list
            ])
            for sim_test_sdr_list in self.sim_test_sdrs
        ]

    def run(self):
        self.print_with_timestamp('==> Run')

        self.test_epoch()
        self.epoch += 1

        n_stages = len(self.data_parts)
        for i in range(n_stages):
            start, n_prototypes = self.data_parts[i]
            masked_sds_range = self.visible_sds_ranges[i]
            self.print_with_timestamp(
                f'Experiment stage {i+1}: {start}:{start+n_prototypes} {masked_sds_range}'
            )
            self.run_stage(
                start=start, n_prototypes=n_prototypes, masked_sds_range=masked_sds_range
            )

        # NB: log last step
        self.log()
        self.print_with_timestamp('<==')

    def run_stage(self, start, n_prototypes, masked_sds_range: tuple[int, int] = None):
        for _ in range(self.n_epochs):
            self.print_with_timestamp(f'Epoch {self.epoch}')
            self.train_epoch(start, n_prototypes, masked_sds_range)
            self.test_epoch()
            self.epoch += 1

    def train_epoch(self, start, n_prototypes, masked_sds_range: tuple[int, int] = None):
        for i_sample in range(self.n_steps):
            prototype = self.data[start + self.rng.choice(n_prototypes)]
            # NB: log just before the next step to include both step and epoch metrics,
            # and also both train and test
            self.log()

            input_sdr = sample_noisy_sdr(self.rng, self.input_sds, prototype, self.noise_level)

            # mask out input sdr to be within a certain range
            if masked_sds_range is not None:
                low, high = masked_sds_range
                input_sdr = input_sdr[(input_sdr >= low) & (input_sdr < high)]

            # print('IN ', input_sdr)
            output_sdr = self.layer.compute(input_sdr, learn=True)
            # print('OUT', output_sdr)
            self.on_step(input_sdr, output_sdr)
            # if i_sample >= 2:
            #     assert False

        self.on_epoch()

    def test_epoch(self):
        for prototype in self.data:
            self.log()
            input_sdr = prototype

            output_sdr = self.layer.compute(prototype, learn=False)
            self.on_step(input_sdr, output_sdr)

        self.on_epoch()

        mod_shift = self.epoch % self.seq_logging_schedule
        if mod_shift in [0, 1]:
            self.metrics |= self.test_sequences()
            self.metrics |= self.test_noisy_pairs()

    def test_sequences(self):
        if not self.logger:
            return {}

        from hima.experiments.temporal_pooling.experiment_stats_tmp import (
            transform_sim_mx_to_plots
        )

        output_seqs = [
            [self.layer.compute(sdr, learn=False) for sdr in seq]
            for seq in self.sim_test_sequences
        ]
        output_mx = self.get_similarity_matrix(output_seqs, self.output_dense_cache)
        abs_err_mx = np.ma.abs(output_mx - self.input_mx)
        diff_dict = dict(
            input_sdr=self.input_mx,
            output_sdr=output_mx,
            abs_err=abs_err_mx,
        )
        sorted_errs = np.sort(abs_err_mx.flatten())
        sorted_errs = sorted_errs[-round(0.1 * len(sorted_errs)):]
        metrics = {
            'sim_mx_diff': diff_dict,
            'sim_mae': abs_err_mx.mean(),
            'sim_mae_top10': sorted_errs.mean(),
        }
        transform_sim_mx_to_plots(metrics)
        return personalize_metrics(metrics, prefix='similarity')

    def test_noisy_pairs(self):
        if not self.logger:
            return {}

        output_avg_sims = [
            np.mean([
                sdr_similarity(
                    self.layer.compute(sdr, learn=False),
                    self.layer.compute(noisy_sdr, learn=False),
                    symmetrical=True, dense_cache=self.output_dense_cache
                )
                for sdr, noisy_sdr in sim_test_sdr_list
            ])
            for sim_test_sdr_list in self.sim_test_sdrs
        ]
        metrics = {
            f'sim_mae_noisy_{round(noise_level*100)}': (
                abs(output_avg_sims[i] - self.input_avg_sims[i])
            )
            for i, noise_level in enumerate(self.sim_noise_level)
        }
        return personalize_metrics(metrics, prefix='similarity')

    def get_similarity_matrix(self, sequences, dense_cache):
        n = len(sequences)
        diagonal_mask = np.identity(n, dtype=bool)
        mx = np.empty((n, n))

        for i in range(n):
            s1 = sequences[i]
            if isinstance(s1[0], int):
                s1 = self.data[s1]

            for j in range(i+1, n):
                s2 = sequences[j]
                if isinstance(s2[0], int):
                    s2 = self.data[s2]

                mx[i, j] = sequence_similarity_elementwise(
                    s1=s1, s2=s2, symmetrical=True, dense_cache=dense_cache
                )
                mx[j, i] = mx[i, j]

        return np.ma.array(mx, mask=diagonal_mask)

    def on_step(self, input_sdr, output_sdr):
        if not self.logger:
            return

        self.metrics |= personalize_metrics(
            metrics=self.input_sdr_tracker.on_sdr_updated(input_sdr, False),
            prefix='input.sdr'
        )
        self.metrics |= personalize_metrics(
            metrics=self.output_sdr_tracker.on_sdr_updated(output_sdr, False),
            prefix='output.sdr'
        )
        self.metrics |= personalize_metrics(
            metrics=self.sp_tracker.on_sp_computed(None, False),
            prefix='sp'
        )

    def on_epoch(self):
        if not self.logger:
            return

        self.metrics |= personalize_metrics(
            metrics=self.input_sdr_tracker.on_sequence_finished(None, False),
            prefix='input.sdr'
        )
        self.metrics |= personalize_metrics(
            metrics=self.output_sdr_tracker.on_sequence_finished(None, False),
            prefix='output.sdr'
        )
        self.metrics |= personalize_metrics(
            metrics=self.sp_tracker.on_sequence_finished(None, False),
            prefix='sp'
        )

    def generate_sim_test_sequences(self):
        sub_sds_list = [
            (
                sds_range[0],
                Sds(size=sds_range[1] - sds_range[0], active_size=self.input_sds.active_size)
            )
            for sds_range in self.visible_sds_ranges
            if sds_range is not None
        ]
        seq_sets = []
        for shift, sub_sds in sub_sds_list:
            base_arr = [
                sample_sdr(self.rng, sub_sds)
                for _ in range(self.n_seq_elements)
            ]

            seq_sets.append([shift + sdr for sdr in base_arr])
            for noise_level in self.sim_noise_level:
                seq_sets.append([
                    shift + sample_noisy_sdr(self.rng, sub_sds, sdr, noise_level)
                    for sdr in base_arr
                ])
            for noise_level in self.sim_noise_level:
                seq_sets.append([
                    sample_noisy_sdr(self.rng, self.input_sds, shift+sdr, noise_level)
                    for sdr in base_arr
                ])
        return seq_sets

    def log(self):
        if not self.logger:
            return

        self.logger.log(self.metrics)
        self.metrics = dict()

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)


def personalize_metrics(metrics: dict, prefix: str):
    return prepend_dict_keys(metrics, prefix, separator='/')
