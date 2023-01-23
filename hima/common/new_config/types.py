#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Type, Callable

TTypeOrFactory = Union[Type, Callable]

TTypeResolver = Union[
    dict[str, TTypeOrFactory],
    'LazyTypeResolver'
]


class LazyTypeResolver(ABC):
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
