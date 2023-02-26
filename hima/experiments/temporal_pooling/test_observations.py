#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from itertools import islice

from hima.common.config.base import TConfig
from hima.common.run.runner import Runner
from hima.common.utils import timed
from hima.experiments.temporal_pooling.data.synthetic_sequences import Sequence
from hima.experiments.temporal_pooling.graph.graph import Pipeline
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.experiment_stats import ExperimentStats
from hima.experiments.temporal_pooling.run_progress import RunProgress
from hima.experiments.temporal_pooling.run_setup import RunSetup
from hima.experiments.temporal_pooling.stats.config import StatsMetricsConfig
from hima.experiments.temporal_pooling.utils import resolve_random_seed, scheduled


class ObservationsLayeredExperiment(Runner):
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
            print(f'Epoch {epoch}: {elapsed_time:.3f} sec')
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

        blocks = self.pipeline.blocks
        sp = blocks['sp2'].sp if 'sp2' in blocks else blocks['sp1']
        print(f'{round(sp.n_computes / sp.run_time / 1000, 2)} kcps')
        # print(.sp.activation_entropy())
        # print('_____')

    def run_sequence(self, sequence: Sequence, i_repeat: int = 0, learn=True):
        self.reset_blocks('temporal_memory', 'temporal_pooler')

        log_scheduled = scheduled(
            i=i_repeat, schedule=self.run_setup.log_repeat_schedule,
            always_report_first=True, always_report_last=True, i_max=self.run_setup.sequence_repeats
        )
        self.stats.on_sequence_started(sequence.id, log_scheduled)

        seq_len = min(len(sequence), self.run_setup.steps_per_sequence)
        for input_data in islice(sequence, seq_len):
            self.reset_blocks('spatial_pooler', 'custom_sp')
            for _ in range(self.run_setup.item_repeats):
                self.progress.next_step()
                self.pipeline.step(input_data, learn=learn)
                self.stats.on_step()

    def reset_blocks(self, *blocks_family):
        blocks_family = set(blocks_family)
        for name in self.pipeline.blocks:
            block = self.pipeline.blocks[name]
            if block.family in blocks_family:
                block.reset()
