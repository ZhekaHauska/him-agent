#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import TYPE_CHECKING

from hima.common.lazy_imports import lazy_import
from hima.common.new_config.base import TConfig
from hima.common.new_config.global_config import GlobalConfig

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
            # we have to pass the config with update instead of init because for sweep runs
            # it is already initialized with the sweep run config
            self.logger.config.update(self.config)

    def run(self) -> None:
        ...


def run(config, config_path, type_resolver):
    global_config = GlobalConfig(config, config_path, type_resolver)

    runner_kwargs = config.copy().update(
        config=config
    )

    # if runner is a callback function, run happens on resolve
    # otherwise, we expect it to be of `Runner` type and should explicitly call `run`
    runner = global_config.object_resolver.resolve(runner_kwargs)
    if isinstance(runner, Runner):
        runner.run()
