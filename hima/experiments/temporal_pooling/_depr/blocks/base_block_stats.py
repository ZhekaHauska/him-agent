#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Any

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


class BlockStats:
    output_sds: Sds

    def __init__(self, output_sds: Sds):
        self.output_sds = output_sds

    def update(self, **kwargs):
        ...

    def step_metrics(self) -> dict[str, Any]:
        return {}

    def final_metrics(self) -> dict[str, Any]:
        ...
