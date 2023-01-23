#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

from hima.common.config.base import TConfig

# ==> special value constants
# They are intended to be non-importable, i.e. to be used only here!


# value should be resolved as None — it is a more explicit way to indicate None than empty string
_TO_BE_NONE_VALUE = '...'

# value must be resolved later by induction on a building stage
#   e.g. layers dimension propagation, DRY principle, calculated dynamically
_TO_BE_INDUCED_VALUE = '???'


def resolve_value(
        value: Any, *,
        substitute_with: Any = _TO_BE_INDUCED_VALUE,
        key: str = None, induction_registry: dict[str, Any] = None
) -> Any:
    """
    Resolve value defined with the config. Some values have specific meaning, which is handled here.
    """
    if value == _TO_BE_NONE_VALUE:
        return None

    elif value == _TO_BE_INDUCED_VALUE:
        # substitute, but default substitution in both scenario — leave it to-be-induced

        if key is None:
            # direct substitution
            return substitute_with
        else:
            # try substitute using registry first then using direct substitution
            return induction_registry.get(key, substitute_with)

    # return already resolved value as is
    return value


# ==================== config dict value induction ====================
def get_unresolved_value() -> Any:
    return _TO_BE_INDUCED_VALUE


def is_resolved_value(value: Any) -> bool:
    return value != _TO_BE_NONE_VALUE and value != _TO_BE_INDUCED_VALUE


def resolve_init_params(config: TConfig, **induction_registry):
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
