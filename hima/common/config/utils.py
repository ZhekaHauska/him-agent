#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from hima.common.config.base import TConfig
from hima.common.config.values import is_resolved_value
from hima.common.sds import Sds, TSdsShortNotation


def resolve_sds(sds: Sds | TConfig):
    sds = try_resolve_sds(sds)
    if isinstance(sds, Sds):
        return sds

    raise ValueError(f'Cannot resolve {sds} as Sds')


def try_resolve_sds(sds: Sds | TConfig | TSdsShortNotation):
    if sds is None:
        # at the moment I don't know of useful SDS interpretation of None
        return None

    if isinstance(sds, Sds):
        return sds

    if not is_resolved_value(sds):
        # allow keeping unresolved values as is, because there's nothing you can do with it RN
        return sds

    return Sds.as_sds(sds)

