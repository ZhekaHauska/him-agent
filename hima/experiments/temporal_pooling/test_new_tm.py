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
from hima.experiments.temporal_pooling.graph.global_vars import (
    VARS_TRACKING_ENABLED, VARS_LEARN,
    VARS_EPOCH, VARS_SEQUENCE_FINISHED, VARS_EPOCH_FINISHED, VARS_STEP, VARS_INPUT,
    VARS_STEP_FINISHED
)
from hima.experiments.temporal_pooling.graph.model import Model
from hima.experiments.temporal_pooling.iteration import IterationConfig
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.run_progress import RunProgress
from hima.experiments.temporal_pooling.utils import resolve_random_seed, scheduled

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


# This is a generalised sequential test.
# It is particularly set up to test new TM memory.
class NewTmExperiment:
    config: GlobalConfig
    logger: Run | None
    init_time: float

    seed: int

    model: Model
    iterate: IterationConfig
    reset_tm: bool

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int,
            iterate: TConfig, reset_tm: bool, data: TConfig,
            model: TConfig,
            stats_and_metrics: TConfig, diff_stats: TConfig,
            log_schedule: TConfig,
            project: str = None,
            wandb_init: TConfig = None,
            track_streams: TConfig = None,
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
        self.reset_tm = reset_tm
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
        self.model.streams[VARS_LEARN].set(True)
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
                print(np.round(np.histogram(ws, bins=10)[0] / stp.output_size, 1))

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
        self.model.streams[VARS_EPOCH].set(self.progress.epoch)
        self.stats.on_epoch_started()

        # if resampling is enabled, epochs are split into stages, where
        # within each stage the set of sequences is fixed
        stage = self.progress.epoch // self.iterate.resample_frequency
        start = stage * self.iterate.sequences
        stop = start + self.iterate.sequences

        for sequence in islice(self.data, start, stop):
            # squash ids to the range [0, self.iterate.sequences)
            real_id = sequence.id
            sequence.id = real_id % self.iterate.sequences

            for i_repeat in range(self.iterate.sequence_repeats):
                self.run_sequence(sequence, i_repeat)
            self.model.streams[VARS_SEQUENCE_FINISHED].set()
            self.stats.on_sequence_finished()

            sequence.id = real_id

        epoch_final_log_scheduled = scheduled(
            i=self.progress.epoch, schedule=self.log_schedule['epoch'],
            always_report_first=True, always_report_last=True, i_max=self.iterate.epochs
        )
        self.model.streams[VARS_EPOCH_FINISHED].set()
        self.stats.on_epoch_finished(epoch_final_log_scheduled)

    def run_sequence(self, sequence: Sequence, i_repeat: int = 0):
        if self.reset_tm:
            self.reset_blocks('temporal_memory', 'temporal_pooler')

        log_scheduled = scheduled(
            i=i_repeat, schedule=self.log_schedule['repeat'],
            always_report_first=True, always_report_last=True, i_max=self.iterate.sequence_repeats
        )
        self.model.streams['sequence_id'].set(sequence.id)
        self.stats.on_sequence_started(sequence.id, log_scheduled)

        for _, input_sdr in enumerate(sequence):
            for _ in range(self.iterate.element_repeats):
                self.progress.next_step()
                self.model.streams[VARS_STEP].set(self.progress.step)
                self.model.metrics.clear()
                self.model.streams[VARS_INPUT].set(input_sdr)

                self.model.forward()

                self.model.streams[VARS_STEP_FINISHED].set()
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
