#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.run.wandb import get_logger
from hima.common.utils import timed
from hima.experiments.temporal_pooling.data.synthetic_sequences import Sequence
from hima.experiments.temporal_pooling.iteration import IterationConfig
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.utils import resolve_random_seed

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class StpExperiment:
    config: GlobalConfig
    logger: Run | None

    seed: int

    iterate: IterationConfig

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, project: str,
            seed: int,
            iterate: TConfig, data: TConfig,
            model: TConfig,
            **_
    ):
        print('==> Init')
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=StpLazyTypeResolver()
        )
        self.logger = get_logger(config, log=log, project=project)
        self.seed = resolve_random_seed(seed)

        self.iterate = self.config.resolve_object(iterate, object_type_or_factory=IterationConfig)
        self.data = self.config.resolve_object(
            data,
            n_sequences=self.iterate.sequences,
            sequence_length=self.iterate.elements
        )
        # block_registry = BlockRegistry(
        #     global_config=self.config, block_configs=blocks,
        #     input_sds=self.data.values_sds
        # )
        # pipeline_resolver = PipelineResolver(block_registry)
        # pipeline = pipeline_resolver.resolve(pipeline)

    def run(self):
        print('==> Run')
        for epoch in range(self.iterate.epochs):
            _, elapsed_time = self.train_epoch()
            print(f'Epoch {epoch}: {elapsed_time:.3f} sec')
        print('<==')

    @timed
    def train_epoch(self):
        # self.progress.next_epoch()
        # self.stats.on_epoch_started()

        # noinspection PyTypeChecker
        for sequence in self.data:
            for i_repeat in range(self.iterate.sequence_repeats):
                self.run_sequence(sequence, i_repeat, learn=True)
            # self.stats.on_sequence_finished()

        # epoch_final_log_scheduled = scheduled(
        #     i=self.progress.epoch, schedule=self.run_setup.log_epoch_schedule,
        #     always_report_first=True, always_report_last=True, i_max=self.run_setup.epochs
        # )
        # self.stats.on_epoch_finished(epoch_final_log_scheduled)

        # blocks = self.pipeline.blocks
        # sp = blocks['sp2'].sp if 'sp2' in blocks else blocks['sp1']
        # print(f'{round(sp.n_computes / sp.run_time / 1000, 2)} kcps')
        # print(.sp.activation_entropy())
        # print('_____')

    def run_sequence(self, sequence: Sequence, i_repeat: int = 0, learn=True):
        # self.reset_blocks('temporal_memory', 'temporal_pooler')

        # log_scheduled = scheduled(
        #     i=i_repeat, schedule=self.run_setup.log_repeat_schedule,
        #     always_report_first=True, always_report_last=True, i_max=self.run_setup.sequence_repeats
        # )
        # self.stats.on_sequence_started(sequence.id, log_scheduled)

        for _, input_sdr in enumerate(sequence):
            # self.reset_blocks('spatial_pooler', 'custom_sp')
            for _ in range(self.iterate.element_repeats):
                # self.progress.next_step()
                # self.pipeline.step(input_data, learn=learn)
                # self.stats.on_step()
                ...
