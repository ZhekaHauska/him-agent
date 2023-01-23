#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Type, Any

from hima.common.config.base import TConfig
from hima.common.config.referencing import ConfigResolver, extracted_type_tag
from hima.common.config.values import resolve_init_params
from hima.common.config.types import TTypeResolver, TTypeOrFactory


class ObjectResolver:
    """
    ObjectResolver is a helper class for resolving and building objects from the configuration.
    """

    type_resolver: TTypeResolver
    config_resolver: ConfigResolver

    def __init__(
            self,
            type_resolver: TTypeResolver = None,
            config_resolver: ConfigResolver = None
    ):
        self.type_resolver = type_resolver
        self.config_resolver = config_resolver

    def resolve(
            self, config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> Any:
        # substitute inducible args using substitution registry
        config = resolve_init_params(config, **substitution_registry)

        if self.config_resolver is not None:
            # we expect that referencing is enabled, so we need to resolve the config
            config = self.config_resolver.resolve(config, config_type=config_type)

        if object_type_or_factory is None:
            # have to resolve the type from the config as object type is not specified
            config, type_tag = extracted_type_tag(config)
            object_type_or_factory = self.type_resolver[type_tag]

        return object_type_or_factory(**config)
