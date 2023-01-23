#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from hima.common.lazy_imports import lazy_import
from hima.common.new_config.base import TConfig
from hima.common.run.wandb import get_logger

wandb = lazy_import('wandb')
if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


# TODO: remove obsolete class and the file itself
class Runner:
    config: TConfig
    config_path: Path
    logger: Run | None

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool = False, project: str = None,
            **unpacked_config
    ):
        warn(DeprecationWarning('This class is obsolete, make your Runner just by convention'))

        self.config = config
        self.config_path = config_path
        self.logger = get_logger(config, log=log, project=project)

    def run(self) -> None:
        ...
