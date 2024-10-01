#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

from hima.common.config.base import TConfig
from hima.common.config.values import is_resolved_value
from hima.common.sds import Sds, TSdsShortNotation


def join_sds(x: Sds | Any, y: Sds | Any) -> Sds:
    if x is None or isinstance(x, Sds):
        return x

    # unresolved -> Sds
    y = make_sds(y)
    if isinstance(y, Sds):
        return y
    return x


def make_sds(sds: Sds | TConfig | TSdsShortNotation | Any) -> Sds | None | Any:
    if sds is None:
        return None

    if isinstance(sds, Sds):
        return sds

    if not is_resolved_value(sds):
        # allow keeping unresolved values as is, because there's nothing you can do with it RN
        return sds

    return Sds.make(sds)
