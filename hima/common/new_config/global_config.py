#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from pathlib import Path
from typing import Union, Dict, Callable, Any

from hima.common.new_config.base import TConfig
from hima.common.new_config.objects import ObjectResolver
from hima.common.new_config.referencing import ConfigResolver
from hima.common.new_config.types import TTypeResolver, LazyTypeResolver


class GlobalConfig:
    config: TConfig
    config_path: Path

    config_resolver: ConfigResolver
    type_resolver: TTypeResolver
    object_resolver: ObjectResolver

    def __init__(self, config: TConfig, config_path: Path, type_resolver: TTypeResolver):
        self.config = config
        self.config_path = config_path

        self.config_resolver = ConfigResolver(
            global_config=config, global_config_path=config_path
        )
        self.type_resolver = type_resolver
        self.object_resolver = ObjectResolver(
            type_resolver=type_resolver, config_resolver=self.config_resolver
        )
