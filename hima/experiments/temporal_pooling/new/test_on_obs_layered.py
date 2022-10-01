#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Optional

import numpy as np
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import TConfig, resolve_value
from hima.common.run_utils import Runner
from hima.common.sds import Sds
from hima.common.utils import timed
from hima.experiments.temporal_pooling.new.run_setup_resolver import (
    resolve_run_setup
)
from hima.experiments.temporal_pooling.new.blocks.dataset_synth_sequences import Sequence
from hima.experiments.temporal_pooling.new.blocks.graph import (
    Block, Pipeline
)
from hima.experiments.temporal_pooling.new.resolvers.graph import (
    PipelineResolver,
    BlockRegistryResolver
)
from hima.experiments.temporal_pooling.new.stats.config import StatsMetricsConfig
from hima.experiments.temporal_pooling.new.experiment_stats import (
    ExperimentStats,
    RunProgress
)


class RunSetup:
    n_sequences: int
    steps_per_sequence: Optional[int]
    sequence_repeats: int
    epochs: int
    log_repeat_schedule: int
    log_epoch_schedule: int

    tp_output_sds: Sds
    sp_output_sds: Sds

    def __init__(
            self, n_sequences: int, steps_per_sequence: Optional[int],
            sequence_repeats: int, epochs: int, total_repeats: int,
            tp_output_sds: Sds.TShortNotation, sp_output_sds: Sds.TShortNotation,
            log_repeat_schedule: int = 1, log_epoch_schedule: int = 1
    ):
        self.n_sequences = n_sequences
        self.steps_per_sequence = steps_per_sequence
        self.sequence_repeats, self.epochs = resolve_epoch_runs(
            sequence_repeats, epochs, total_repeats
        )
        self.log_repeat_schedule = log_repeat_schedule
        self.log_epoch_schedule = log_epoch_schedule

        self.tp_output_sds = Sds(short_notation=tp_output_sds)
        self.sp_output_sds = Sds(short_notation=sp_output_sds)


class ObservationsLayeredExperiment(Runner):
    config: TConfig
    logger: Optional[Run]

    seed: int
    run_setup: RunSetup
    stats_config: StatsMetricsConfig
    pipeline: Pipeline
    generator: Block
    progress: RunProgress
    stats: ExperimentStats

    normalization_unbias: str

    def __init__(
            self, config: TConfig, seed: int, debug: bool,
            blocks: dict[str, TConfig], pipeline: list,
            run_setup: TConfig, stats_and_metrics: TConfig,
            track_streams: TConfig, diff_stats: TConfig,
            loss: list[str], charts: list[str],
            **_
    ):
        super().__init__(config, **config)
        print('==> Init')

        self.seed = resolve_random_seed(seed)
        self.run_setup = resolve_run_setup(config, run_setup, experiment_type='layered')
        self.stats_config = StatsMetricsConfig(**stats_and_metrics)

        self.pipeline = PipelineResolver(
            block_registry=BlockRegistryResolver(
                config=config, block_configs=blocks,
                seed=seed,
                n_sequences=self.run_setup.n_sequences,
            )
        ).resolve(pipeline)
        print(self.pipeline.blocks)
        print(self.pipeline)

        self.generator = self.pipeline.blocks['gen']
        self.progress = RunProgress()
        self.stats = ExperimentStats(
            n_sequences=self.run_setup.n_sequences,
            progress=self.progress, logger=self.logger,
            blocks=self.pipeline.blocks, track_streams=track_streams,
            stats_config=self.stats_config,
            debug=debug, diff_stats=diff_stats,
            loss=loss, charts=charts
        )

    def run(self):
        print('==> Run')
        self.stats.define_metrics()

        for epoch in range(self.run_setup.epochs):
            _, elapsed_time = self.train_epoch()
            print(f'Epoch {epoch}: {elapsed_time}')
        print('<==')

    @timed
    def train_epoch(self):
        self.progress.next_epoch()
        self.stats.on_epoch_started()

        # noinspection PyTypeChecker
        for sequence in self.generator:
            for i_repeat in range(self.run_setup.sequence_repeats):
                self.run_sequence(sequence, i_repeat, learn=True)
            self.stats.on_sequence_finished()

        epoch_final_log_scheduled = scheduled(
            i=self.progress.epoch, schedule=self.run_setup.log_epoch_schedule,
            always_report_first=True, always_report_last=True, i_max=self.run_setup.epochs
        )
        self.stats.on_epoch_finished(epoch_final_log_scheduled)

    def run_sequence(self, sequence: Sequence, i_repeat: int = 0, learn=True):
        self.reset_blocks('temporal_memory', 'temporal_pooler')

        log_scheduled = scheduled(
            i=i_repeat, schedule=self.run_setup.log_repeat_schedule,
            always_report_first=True, always_report_last=True, i_max=self.run_setup.sequence_repeats
        )
        self.stats.on_sequence_started(sequence.id, log_scheduled)

        for input_data in sequence:
            self.progress.next_step()
            self.pipeline.step(input_data, learn=learn)
            self.stats.on_step()

    def reset_blocks(self, *blocks_family):
        blocks_family = set(blocks_family)
        for name in self.pipeline.blocks:
            block = self.pipeline.blocks[name]
            if block.family in blocks_family:
                block.reset()


def resolve_random_seed(seed: Optional[int]) -> int:
    seed = resolve_value(seed)
    if seed is None:
        # randomly generate a seed
        return np.random.default_rng().integers(10000)
    return seed


def resolve_epoch_runs(intra_epoch_repeats: int, epochs: int, total_repeats: int):
    total_repeats = resolve_value(total_repeats)
    intra_epoch_repeats = resolve_value(intra_epoch_repeats)
    epochs = resolve_value(epochs)
    if intra_epoch_repeats is None:
        intra_epoch_repeats = total_repeats // epochs
    if epochs is None:
        epochs = total_repeats // intra_epoch_repeats
    return intra_epoch_repeats, epochs


def scheduled(
        i: int, schedule: int = 1,
        always_report_first: bool = True, always_report_last: bool = True, i_max: int = None
):
    if always_report_first and i == 0:
        return True
    if always_report_last and i + 1 == i_max:
        return True
    if (i + 1) % schedule == 0:
        return True
    return False
