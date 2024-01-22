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
    sample_random_sdr,
    sample_noisy_sdr
)
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
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
            noise_level: float,
            n_epochs: int,
            n_steps: int,

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

        self.n_prototypes = n_prototypes
        self.noise_level = noise_level
        self.n_epochs = n_epochs
        self.n_steps = n_steps

        self.data = [
            sample_random_sdr(self.rng, self.input_sds)
            for _ in range(2 * self.n_prototypes)
        ]

        self.layer = self.config.resolve_object(
            layer, feedforward_sds=self.input_sds, output_sds=self.output_sds,
        )

        sdr_tracker_config = dict(step_flush_schedule=100, aggregate_flush_schedule=100)
        self.input_sdr_tracker = SdrTracker(self.input_sds, **sdr_tracker_config)
        self.output_sdr_tracker = SdrTracker(self.output_sds, **sdr_tracker_config)
        self.sp_tracker = SpTracker(self.layer, **sdr_tracker_config)

        self.epoch = 0
        self.metrics = dict()

    def run(self):
        self.print_with_timestamp('==> Run')

        self.test_epoch()
        self.epoch += 1

        self.train_epoch(start=0, n_prototypes=self.n_prototypes)
        self.test_epoch()
        self.train_epoch(start=self.n_prototypes, n_prototypes=self.n_prototypes)
        self.test_epoch()
        self.train_epoch(start=0, n_prototypes=2*self.n_prototypes)
        self.test_epoch()

        # NB: log last step
        self.log()
        self.print_with_timestamp('<==')

    def train_epoch(self, start, n_prototypes):
        self.print_with_timestamp(f'Epoch {self.epoch}')
        for i_sample in range(self.n_steps):
            prototype = self.data[start + self.rng.choice(n_prototypes)]
            # NB: log just before the next step to include both step and epoch metrics,
            # and also both train and test
            self.log()

            input_sdr = sample_noisy_sdr(self.rng, self.input_sds, prototype, self.noise_level)
            output_sdr = self.layer.compute(input_sdr, learn=True)

            self.on_step(input_sdr, output_sdr)

        self.on_epoch()
        self.epoch += 1

    def test_epoch(self):
        for prototype in self.data:
            self.layer.compute(prototype, learn=False)

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
