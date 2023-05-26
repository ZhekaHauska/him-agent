#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from hima.common.lazy_imports import lazy_import
from hima.common.config.base import TConfig
from hima.common.utils import isnone

wandb = lazy_import('wandb')
if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


def turn_off_gui_for_matplotlib():
    """
    Prevent matplotlib from using any GUI backend.
    For example, it is prohibited for sub-processes as you will encounter kernel core errors.
    """
    from matplotlib import pyplot as plt
    plt.switch_backend('Agg')


def set_wandb_sweep_threading():
    # on Linux machines there's some kind of problem with running sweeps in threads?
    # see https://github.com/wandb/client/issues/1409#issuecomment-870174971
    # and https://github.com/wandb/client/issues/3045#issuecomment-1010435868
    os.environ['WANDB_START_METHOD'] = 'thread'


def set_wandb_entity(entity):
    # overwrite wandb entity for the run
    os.environ['WANDB_ENTITY'] = entity


class DryWandbLogger:
    def __init__(self, *_, **__):
        pass

    def __getattr__(self, item):
        return lambda *_, **__: None

    def log(self, *_, **__):
        pass


def get_logger(
        config: TConfig, log: bool | str | None, project: str = None, wandb_init: TConfig = None
) -> Run | None:
    if log is None or not log:
        return None

    if log == 'dry':
        # imitate wandb logger, but do nothing => useful for debugging
        # noinspection PyTypeChecker
        return DryWandbLogger()

    wandb_init = isnone(wandb_init, {})
    logger = wandb.init(project=project, **wandb_init)

    # we have to pass the config with update instead of init because for sweep runs
    # it is already initialized with the sweep run config
    logger.config.update(config)
    return logger


