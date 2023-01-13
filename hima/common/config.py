#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any, Collection

from hima.common.utils import ensure_list

# Register config-related conventional constants here.
# NB: They are intended to be non-importable, i.e. to be used only here!
_TYPE_KEY = '_type_'
_TYPE_FAMILY_KEY = '_type_family_'
_BASE_CONFIG_KEY = '_base_config_'
_TO_BE_NONE_VALUE = '...'
_TO_BE_INDUCED_VALUE = '???'

TConfig = dict[str, Any]
TKeyPathValue = tuple[list, Any]


# ==================== config dict compilation and values parsing ====================
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


# ==================== config dict slicing ====================
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
    which is tuple[TConfig, Optional[Any], ...], so less strict type hint is provided

    TODO: Probably, it should be reworked to a `split`-like function that returns two dicts

    Examples
    --------
    >>> extracted({'a': 1, 'b': 2, 'c': 3}, 'a', 'c')
    ({'b': 2}, 1, 3)
    """
    values = tuple([d.get(k, None) for k in keys])
    filtered_dict = filtered(d, keys, depth=1)
    return (filtered_dict, ) + values


# ==================== config meta info extraction ====================
def extracted_type(config: TConfig) -> tuple[TConfig, str | None]:
    """Extracts the type using the type hinting convention for configs."""
    return extracted(config, _TYPE_KEY)


def extracted_base_config(config: TConfig) -> tuple[TConfig, str | None]:
    """Extracts the base config name using the meta key convention for configs."""
    return extracted(config, _BASE_CONFIG_KEY)


def extracted_family(config: TConfig) -> tuple[TConfig, str | None]:
    """Extracts the type family using the meta key convention for configs."""
    return extracted(config, _TYPE_FAMILY_KEY)


# ==================== resolve nested configs ====================
def resolve_nested_configs(
        config_registry: TConfig, *,
        config_name: str = None, config: TConfig = None
) -> TConfig:
    # if config itself is not provided, we need to resolve it by name from the registry
    if config is None:
        if config_name is None:
            return {}
        config = config_registry[config_name]

    config, base_config_name = extracted_base_config(config)
    if base_config_name is not None:
        # recursively build base configs starting from the innermost one
        base_config = resolve_nested_configs(config_registry, config_name=base_config_name)
        # TODO: it may require unusual dict merge logic for the special values
        base_config.update(**config)
        config = base_config

    return config


# ==================== config dict value induction ====================
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


def get_none_value() -> Any:
    return _TO_BE_NONE_VALUE


def get_unresolved_value() -> Any:
    return _TO_BE_INDUCED_VALUE


def is_resolved_value(value: Any) -> bool:
    return value != _TO_BE_NONE_VALUE and value != _TO_BE_INDUCED_VALUE


def resolve_value(
        value: Any, *,
        substitute_with: Any = None,
        key: str = None, induction_registry: dict = None
) -> Any:
    """
    Resolve value defined with the config. Some values have specific meaning, which is handled here.
    """
    if value == _TO_BE_NONE_VALUE:
        return None
    elif value == _TO_BE_INDUCED_VALUE:
        if key is None:
            # direct substitution
            return substitute_with
        else:
            # try substitute using registry or leave it unchanged
            return induction_registry.get(key, _TO_BE_INDUCED_VALUE)
    return value


def resolve_init_params(config: dict, **induction_registry):
    """
    Resolve params defined with the config. Some values are intended to be resolved
    later at runtime - so, it tries to substitute special values with the
    values from the induction registry.
    """
    return {
        k: resolve_value(config[k], key=k, induction_registry=induction_registry)
        for k in config
    }


def check_all_resolved(*values) -> bool:
    """Check all provided values are resolved, i.e. there is no value equal to specific constant"""
    resolved = True
    for x in values:
        resolved &= is_resolved_value(x)
    return resolved
