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
from hima.experiments.temporal_pooling.resolvers.graph import (
    BlockRegistry,
    PipelineResolver
)
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.utils import resolve_random_seed

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class StpExperiment:
    config: GlobalConfig
    logger: Run | None

    seed: int

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, project: str,
            seed: int,
            data: TConfig, pipeline: TConfig, blocks: TConfig,
            **_
    ):
        print('==> Init')
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=StpLazyTypeResolver()
        )
        self.logger = get_logger(config, log=log, project=project)
        self.seed = resolve_random_seed(seed)
        self.data = self.config.resolve_object(
            data,
            seed=self.seed, n_sequences=10
        )
        # block_registry = BlockRegistry(
        #     global_config=self.config, block_configs=blocks,
        #     input_sds=self.data.values_sds
        # )
        # pipeline_resolver = PipelineResolver(block_registry)
        # pipeline = pipeline_resolver.resolve(pipeline)

    def run(self):
        print('==> Run')
        print('<==')
