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
from hima.common.sds import TSdsShortNotation, Sds
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


class NeurogenesisExperiment:
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
            binary: bool,
            input_sds: TSdsShortNotation,
            output_sds: TSdsShortNotation,
            n_prototypes: int,
            noise_level: float,
            n_samples: int,

            layer: TConfig,

            # iterate: TConfig, data: TConfig,
            # model: TConfig,
            # log_schedule: TConfig,
            project: str = None,
            wandb_init: TConfig = None,
            # track_streams: TConfig = None,
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

        input_sds = Sds.make(input_sds)
        output_sds = Sds.make(output_sds)

        self.data = []

        self.layer = self.config.resolve_object(
            layer, feedforward_sds=input_sds, output_sds=output_sds,
        )

    def run(self):
        self.print_with_timestamp('==> Run')

        for sdr in self.data:
            self.layer.compute(sdr)

        self.print_with_timestamp('<==')

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)
