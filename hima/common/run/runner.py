#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import TYPE_CHECKING

from hima.common.lazy_imports import lazy_import

from hima.common.config import TConfig


wandb = lazy_import('wandb')
if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class Runner:
    config: TConfig
    logger: Run | None

    def __init__(
            self, config: TConfig, log: bool = False, project: str = None,
            **unpacked_config
    ):
        self.config = config
        self.logger = None
        if log:
            self.logger = wandb.init(project=project)
            # we have to pass the config with update instead of init because of sweep runs
            self.logger.config.update(self.config)

    def run(self) -> None:
        ...
