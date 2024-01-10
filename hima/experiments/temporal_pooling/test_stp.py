#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.run.wandb import get_logger
from hima.common.timer import timer, print_with_timestamp
from hima.common.utils import timed, isnone
from hima.experiments.temporal_pooling.data.synthetic_sequences import Sequence
from hima.experiments.temporal_pooling.graph.global_vars import VARS_TRACKING_ENABLED
from hima.experiments.temporal_pooling.graph.model import Model
from hima.experiments.temporal_pooling.iteration import IterationConfig
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.run_progress import RunProgress
from hima.experiments.temporal_pooling.utils import resolve_random_seed, scheduled

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


# This is a generalised sequential test.
# It tests sequential memory on different sequential datasets.
class StpExperiment:
    config: GlobalConfig
    logger: Run | None
    init_time: float

    seed: int

    model: Model
    iterate: IterationConfig

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int,
            iterate: TConfig, data: TConfig,
            model: TConfig,
            track_streams: TConfig, stats_and_metrics: TConfig, diff_stats: TConfig,
            log_schedule: TConfig,
            project: str = None,
            wandb_init: TConfig = None,
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

        self.iterate = self.config.resolve_object(iterate, object_type_or_factory=IterationConfig)
        self.data = self.config.resolve_object(
            data,
            n_sequences=self.iterate.total_sequences,
            sequence_length=self.iterate.elements
        )
        self.model = self.config.resolve_object(model, object_type_or_factory=Model)

        # propagate data SDS to the model graph
        self.model.streams['input.sdr'].set_sds(self.data.sds)
        self.model.compile()
        print(self.model)
        print()

        self.progress = RunProgress()
        from hima.experiments.temporal_pooling._depr.stats.config import StatsMetricsConfig
        stats_and_metrics = self.config.resolve_object(
            stats_and_metrics, object_type_or_factory=StatsMetricsConfig
        )
        from hima.experiments.temporal_pooling.experiment_stats_tmp import ExperimentStats
        self.stats = ExperimentStats(
            n_sequences=self.iterate.sequences, progress=self.progress, logger=self.logger,
            model=self.model, track_streams=track_streams, stats_config=stats_and_metrics,
            diff_stats=diff_stats
        )
        self.log_schedule = log_schedule

    def run(self):
        self.print_with_timestamp('==> Run')
        self.stats.define_metrics()
        self.model.streams[VARS_TRACKING_ENABLED].set(self.logger is not None)

        from hima.experiments.temporal_pooling.stp.stp import SpatialTemporalPooler
        import numpy as np
        stp = None
        if 'STE' in self.model.blocks:
            stp: SpatialTemporalPooler = self.model.blocks['STE'].sp
        elif 'SE' in self.model.blocks:
            stp: SpatialTemporalPooler = self.model.blocks['SE'].sp

        for epoch in range(self.iterate.epochs):
            self.print_with_timestamp(f'Epoch {epoch}')
            _, elapsed_time = self.train_epoch()

            if stp is not None:
                if isinstance(stp, SpatialTemporalPooler):
                    indices = np.arange(stp.rf_size)
                    mask = (indices < stp.corrected_rf_size).flatten()
                    ws = stp.weights.flatten()[mask]
                else:
                    ws = stp.weights
                print(np.mean(ws), np.std(ws))
                print(np.round(np.histogram(ws, bins=20)[0] / stp.output_size, 1))

        if self.logger:
            try:
                self.logger.config.update(self.config.config, allow_val_change=True)
            except:
                # quick-n-dirty hack to remedy DryWandbLogger's inability to do this :)
                pass
        self.print_with_timestamp('<==')

    @timed
    def train_epoch(self):
        self.progress.next_epoch()
        self.model.streams['epoch'].set(self.progress.epoch)
        self.stats.on_epoch_started()

        # TODO: FINISH
        if self.iterate.resample_frequency < self.iterate.epochs:
            self.train_epoch_with_switch_data()
            return

        # noinspection PyTypeChecker
        for sequence in self.data:
            for i_repeat in range(self.iterate.sequence_repeats):
                self.run_sequence(sequence, i_repeat, learn=True)
            self.model.streams['sequence_finished'].set()
            self.stats.on_sequence_finished()

        epoch_final_log_scheduled = scheduled(
            i=self.progress.epoch, schedule=self.log_schedule['epoch'],
            always_report_first=True, always_report_last=True, i_max=self.iterate.epochs
        )
        self.model.streams['epoch_finished'].set()
        self.stats.on_epoch_finished(epoch_final_log_scheduled)

        # blocks = self.pipeline.blocks
        # sp = blocks['sp2'].sp if 'sp2' in blocks else blocks['sp1']
        # print(f'{round(sp.n_computes / sp.run_time / 1000, 2)} kcps')
        # print(.sp.activation_entropy())
        # print('_____')

    def train_epoch_with_switch_data(self):
        stage = self.progress.epoch // self.iterate.resample_frequency
        start = stage * self.iterate.sequences
        stop = start + self.iterate.sequences

        # noinspection PyTypeChecker
        for sequence in islice(self.data, start, stop):
            # HACK
            sequence.id = sequence.id % self.iterate.sequences

            for i_repeat in range(self.iterate.sequence_repeats):
                self.run_sequence(sequence, i_repeat, learn=True)
            self.stats.on_sequence_finished()

        epoch_final_log_scheduled = scheduled(
            i=self.progress.epoch, schedule=self.log_schedule['epoch'],
            always_report_first=True, always_report_last=True, i_max=self.iterate.epochs
        )
        self.stats.on_epoch_finished(epoch_final_log_scheduled)

    def run_sequence(self, sequence: Sequence, i_repeat: int = 0, learn=True):
        self.reset_blocks('temporal_memory', 'temporal_pooler')

        log_scheduled = scheduled(
            i=i_repeat, schedule=self.log_schedule['repeat'],
            always_report_first=True, always_report_last=True, i_max=self.iterate.sequence_repeats
        )
        self.stats.on_sequence_started(sequence.id, log_scheduled)
        self.model.streams['sequence_id'].set(sequence.id)

        for _, input_sdr in enumerate(sequence):
            for _ in range(self.iterate.element_repeats):
                self.progress.next_step()
                self.model.streams['step'].set(self.progress.step)
                self.model.metrics.clear()
                self.model.streams['input.sdr'].set(input_sdr)

                self.model.forward()

                self.model.streams['step_finished'].set()
                self.stats.on_step()

    def reset_blocks(self, *blocks_family):
        blocks_family = set(blocks_family)
        for name in self.model.blocks:
            block = self.model.blocks[name]
            if block.family in blocks_family:
                block.reset()

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)
