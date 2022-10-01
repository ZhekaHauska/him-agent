#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any


def rename_dict_keys(d: dict[str, Any], add_prefix):
    return {
        f'{add_prefix}{k}': d[k]
        for k in d
    }
