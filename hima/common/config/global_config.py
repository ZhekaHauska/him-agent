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


# TODO:
#  - allow multiple base configs. Allow overrides that override entirely the list or append to it
#  - improve working with external imports (from another files) and relative imports
#  - allow recursive config resolution/substitution (e.g. merge nested dicts with resolution)
#  - allow local base configs (they need to be masked out)
#  - specify separation of configs: raw base config + overrides, constructed resolved namespace,
#       visible resolved configs (if it is needed). Which one should be logged?
#  - solve problem with dynamic overrides order (before or after base config resolution)
#  - decide, whether we should dynamically induce __init__ params from an object to
#       understand whether or not it needs common params (seed, global_config, etc.),
#       and whether we want them to be resolved implicitly or explicitly


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
            self, _config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> Any:
        """
        Return a new object constructed from its config.
        It requires resolving an object's requirements — its config and factory method.
        """
        return self.object_resolver.resolve(
            _config,
            object_type_or_factory=object_type_or_factory,
            config_type=config_type,
            **substitution_registry | self.global_substitution_registry
        )

    def resolve_object_requirements(
            self, _config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> tuple[TConfig, TTypeOrFactory]:
        """
        Resolve and return requirements for construction a new object w/o constructing it.
        """
        return self.object_resolver.resolve_requirements(
            _config,
            object_type_or_factory=object_type_or_factory,
            config_type=config_type,
            **substitution_registry | self.global_substitution_registry
        )
