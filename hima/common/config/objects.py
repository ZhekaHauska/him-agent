#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Type, Any

from hima.common.config.base import TConfig
from hima.common.config.referencing import ConfigResolver, extracted_type_tag
from hima.common.config.types import TTypeResolver, TTypeOrFactory
from hima.common.config.values import resolve_init_params, is_resolved_value


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

    def resolve_requirements(
            self, _config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> tuple[TConfig, TTypeOrFactory]:
        if not is_resolved_value(_config) or _config is None:
            raise ValueError(f'{_config}')

        if self.config_resolver is not None:
            # we expect that referencing is enabled, so we need to resolve the config
            _config = self.config_resolver.resolve(_config, config_type=config_type)

        if config_type is dict:
            # substitute inducible args using substitution registry
            _config = resolve_init_params(_config, **substitution_registry)

        if object_type_or_factory is None:
            # have to resolve the type from the config as object type is not specified
            _config, type_tag = extracted_type_tag(_config)
            object_type_or_factory = self.type_resolver[type_tag]

        return _config, object_type_or_factory

    def resolve(
            self, _config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> Any:
        try:
            _config, object_type_or_factory = self.resolve_requirements(
                _config, object_type_or_factory=object_type_or_factory,
                config_type=config_type, **substitution_registry
            )
            if config_type is list:
                return object_type_or_factory(*_config)
            return object_type_or_factory(**_config)
        except (TypeError, AttributeError):
            from pprint import pprint
            pprint(_config)
            pprint(substitution_registry)
            print(f'object_type_or_factory: {object_type_or_factory} | config_type: {config_type}')
            raise
