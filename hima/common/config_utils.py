#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from ast import literal_eval
from pathlib import Path
from typing import Iterable, Any, Optional, Union

from ruamel import yaml

from hima.common.utils import ensure_list

# TODO: rename file to `config` and add top-level description comment


# Register config-related conventional constants here.
# NB: They are intended to be non-importable, i.e. to be used only here!
_TYPE_KEY = '_type_'
_TYPE_FAMILY_KEY = '_type_family_'
_BASE_CONFIG_KEY = '_base_config_'
_TO_BE_NONE_VALUE = '...'
_TO_BE_INDUCED_VALUE = '???'


TConfig = dict[str, Any]
TConfigOverrideKV = tuple[list, Any]


# ==================== config dict slicing ====================
def filtered(d: TConfig, keys_to_remove: Iterable[str], depth: int) -> TConfig:
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
def extracted_type(config: TConfig) -> tuple[TConfig, Optional[str]]:
    """Extracts the type using the type hinting convention for configs."""
    return extracted(config, _TYPE_KEY)


def extracted_base_config(config: TConfig) -> tuple[TConfig, Optional[str]]:
    """Extracts the base config name using the meta key convention for configs."""
    return extracted(config, _BASE_CONFIG_KEY)


def extracted_family(config: TConfig) -> tuple[TConfig, Optional[str]]:
    """Extracts the type family using the meta key convention for configs."""
    return extracted(config, _TYPE_FAMILY_KEY)


def extracted_meta_info(config: TConfig) -> tuple[
    TConfig, Optional[str], Optional[str], Optional[str]
]:
    """
    Extracts 1) the type family, 2) concrete type and 3) link to the base config
    using the meta key convention for configs.
    (see private const keys in the config_utils.py)
    """
    return extracted(config, _TYPE_FAMILY_KEY, _TYPE_KEY, _BASE_CONFIG_KEY)


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

    # recursively build base configs starting from the innermost one
    config, base_config_name = extracted_base_config(config)
    if base_config_name is not None:
        base_config = resolve_nested_configs(config_registry, config_name=base_config_name)
        # TODO: it may require unusual dict merge logic for the special values
        base_config.update(**config)
        config = base_config

    return config


# ==================== config dict value induction ====================
def resolve_absolute_quantity(abs_or_relative: Union[int, float], baseline: int) -> int:
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


def resolve_relative_quantity(abs_or_relative: Union[int, float], baseline: int) -> float:
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


# ==================== config dict compilation and values parsing ====================
def override_config(
        config: TConfig,
        overrides: Union[TConfigOverrideKV, list[TConfigOverrideKV]]
) -> None:
    """Applies the number of overrides to the content of the config dictionary."""
    overrides = ensure_list(overrides)
    for key_path, value in overrides:
        c = config
        for key_token in key_path[:-1]:
            c = c[key_token]
        c[key_path[-1]] = value


def parse_arg(arg: Union[str, tuple[str, Any]]) -> TConfigOverrideKV:
    if isinstance(arg, str):
        # raw arg string: "key=value"

        # "--key=value" --> ["--key", "value"]
        key_path, value = arg.split('=', maxsplit=1)

        # "--key" --> "key"
        key_path = key_path.removeprefix('--')

        # parse value represented as str
        value = parse_str(value)
    else:
        # tuple ("key", value) from wandb config of the sweep single run
        # we assume that the value is already passed correctly parsed
        key_path, value = arg

    # We parse key tokens as they can represent array indices
    # We skip empty key tokens (see [1] in the end of the file for an explanation)
    key_path = [
        parse_str(key_token)
        for key_token in key_path.split('.')
        if key_token
    ]

    return key_path, value


def parse_str(s: str) -> Any:
    """Parse string value to the most appropriate type."""

    # noinspection PyShadowingNames
    def boolify(s):
        if s in ['True', 'true']:
            return True
        if s in ['False', 'false']:
            return False
        raise ValueError('Not Boolean Value!')

    # NB: try/except is widely accepted pythonic way to parse things
    assert isinstance(s, str)

    # NB: order here is important
    for caster in (boolify, int, float, literal_eval):
        try:
            return caster(s)
        except ValueError:
            pass
    return s


def read_config(filepath: str) -> TConfig:
    filepath = Path(filepath)
    with filepath.open('r') as config_io:
        return yaml.load(config_io, Loader=yaml.Loader)


# [1]: Using sweeps we have a little problem with config logging. All parameters
# provided to a run from sweep are logged to wandb automatically. At the same time, when
# we also log our compiled config dictionary, its content is flattened such that
# each param key is represented as `path.to.nested.dict.key`. Note that we declare
# params in sweep config the same way. Therefore, each sweep run will have such params
# duplicated in wandb and there's no correct way to distinguish them. However, wandb
# does it! Also, only sweep runs will have params duplicated. Simple runs don't have
# the second entry because they don't have sweep param args.
#
# Problem: when you want to filter or group by param in wandb interface,
# you cannot be sure which of the duplicated entries to choose, while they're different
# — the only entry that is presented in all runs [either sweep or simple] is the entry
# from our config, not from a sweep.
#
# Solution: That's why we introduced a trick - it's allowed to specify sweep param
# with insignificant additional dots (e.g. `path..to...key.`) to de-duplicate entries.
# We ignore these dots [or empty path elements introduced by them after split-by-dots]
# while parsing the nested key path.
