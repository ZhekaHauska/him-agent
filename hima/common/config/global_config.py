#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from hima.common.config.base import TConfig
from hima.common.config.objects import ObjectResolver
from hima.common.config.referencing import ConfigResolver
from hima.common.config.types import TTypeResolver, TTypeOrFactory


class GlobalConfig:
    config: TConfig
    config_path: Path

    config_resolver: ConfigResolver
    type_resolver: TTypeResolver
    object_resolver: ObjectResolver

    global_substitution_registry: dict

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
        self.global_substitution_registry = dict(
            global_config=self,
            seed=self.config['seed'],
        )

    def resolve_object(
            self, config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> Any:
        return self.object_resolver.resolve(
            config,
            object_type_or_factory=object_type_or_factory,
            config_type=config_type,
            global_config=self,
            **substitution_registry | self.global_substitution_registry
        )
