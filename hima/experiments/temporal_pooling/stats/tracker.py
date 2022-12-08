#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

from hima.common.sdr import SparseSdr

TMetrics = dict[str, Any]


class Tracker:
    def on_sequence_started(self, sequence_id: int):
        pass

    def on_epoch_started(self):
        pass

    def on_step(self, sdr: SparseSdr):
        pass

    def on_sequence_finished(self):
        pass

    def on_epoch_finished(self):
        pass

    def step_metrics(self) -> TMetrics:
        return {}

    def aggregate_metrics(self) -> TMetrics:
        return {}
