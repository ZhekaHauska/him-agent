#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any, Collection, Union

from hima.common.utils import ensure_list


# config-related types
TConfig = Union[
    dict[str, Any],
    list[Any]
]
TKeyPath = list
TKeyPathValue = tuple[TKeyPath, Any]


# ==================== resolve absolute or relative quantity ====================
# quantities can be specified as absolute or relative to some baseline value

def resolve_absolute_quantity(abs_or_relative: int | float, *, baseline: int) -> int:
    """
    Convert passed quantity to the absolute quantity regarding its type and the baseline value.
    Here we consider that ints relate to the absolute quantities and floats
    relate to the relative quantities (relative to the `baseline` value).

    Examples:
        ensure_absolute(10, 20) -> 10
        ensure_absolute(1.25, 20) -> 25


    Parameters
    ----------
    abs_or_relative: int or float
        The value to convert. If it's int then it's returned as is. Otherwise, it's
        converted to the absolute system relative to the `baseline` value
    baseline: int
        The baseline for the relative number system.

    Returns
    -------
        Integer value in the absolute quantities system
    """

    if isinstance(abs_or_relative, float):
        relative = abs_or_relative
        return int(baseline * relative)
    elif isinstance(abs_or_relative, int):
        absolute = abs_or_relative
        return absolute
    else:
        raise TypeError(f'Function does not support type {type(abs_or_relative)}')


def resolve_relative_quantity(abs_or_relative: int | float, *, baseline: int) -> float:
    """See `resolve_absolute_quantity` - this method is the opposite of it."""

    if isinstance(abs_or_relative, float):
        relative = abs_or_relative
        return relative
    elif isinstance(abs_or_relative, int):
        absolute = abs_or_relative
        return absolute / baseline
    else:
        raise TypeError(f'Function does not support type {type(abs_or_relative)}')


# ==================== config dict manipulation utils ====================
def override_config(
        config: TConfig,
        overrides: list[TKeyPathValue] | TKeyPathValue
) -> None:
    """Apply the number of overrides to the content of the config dictionary."""
    overrides = ensure_list(overrides)
    for key_path, value in overrides:
        c = config
        for key_token in key_path[:-1]:
            c = c[key_token]
        c[key_path[-1]] = value


def filtered(d: TConfig, keys_to_remove: Collection[str], depth: int) -> TConfig:
    """
    Return a shallow copy of the provided dictionary without the items
    that match `keys_to_remove`.

    The `depth == 1` means filtering `d` itself,
        `depth == 2` — with its dict immediate descendants
        and so on.
    """
    if not isinstance(d, dict) or depth <= 0:
        return d

    return {
        k: filtered(v, keys_to_remove, depth - 1)
        for k, v in d.items()
        if k not in keys_to_remove
    }


def extracted(d: TConfig, *keys: str) -> tuple:
    """
    Return a copy of the dictionary without specified keys and each extracted value
    (or None if a specified key was absent).

    NOTE: Sadly, type checkers incorrectly understand the correct type hint here,
    which is tuple[TConfig, Optional[Any], ...], so a less strict type hint is provided

    Examples
    --------
    >>> extracted({'a': 1, 'b': 2, 'c': 3}, 'a', 'c')
    ({'b': 2}, 1, 3)
    """
    values = tuple([d.get(k, None) for k in keys])
    filtered_dict = filtered(d, keys, depth=1)
    return (filtered_dict, ) + values
