#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, Any, Callable, Union

from hima.common.new_config.base import TConfig
from hima.common.new_config.referencing import ConfigResolver, extracted_type_tag
from hima.common.new_config.value_induction import resolve_init_params

TTypeOrFactory = Union[
    Type,
    Callable[..., Type]
]


class TypeResolverBase(ABC):
    """
    TypeResolverBase is a convenient base class for all type resolvers.
    It is a dictionary-like object that lazily resolves type tags to types.
    """
    types: dict[str, TTypeOrFactory]

    def __init__(self):
        self.types = {}

    def __getitem__(self, type_tag: str) -> TTypeOrFactory:
        resolved_type = self.types.get(type_tag)

        if resolved_type is None:
            # lazy loading of types prevents unused imports
            resolved_type = self.resolve(type_tag)
            self.types[type_tag] = resolved_type

        return resolved_type

    @abstractmethod
    def resolve(self, type_tag: str) -> TTypeOrFactory:
        """Returns the type of object by its type tag."""
        raise NotImplementedError()


class ObjectResolver:
    """
    ObjectResolver is a helper class for resolving and building objects from the configuration.
    """

    config_resolver: ConfigResolver
    type_resolver: TypeResolverBase

    def __init__(self, config_resolver: ConfigResolver, type_resolver: TypeResolverBase):
        self.config_resolver = config_resolver
        self.type_resolver = type_resolver

    def resolve(self, config: TConfig, **substitution_registry) -> Any:
        # substitute inducible args using substitution registry
        config = resolve_init_params(config, **substitution_registry)

        # extract type tag and then resolve object builder
        config, type_tag = extracted_type_tag(config)
        object_type_or_factory = self.type_resolver[type_tag]

        return object_type_or_factory(**config)

