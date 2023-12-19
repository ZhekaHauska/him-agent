#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.experiments.temporal_pooling.stats.metrics import TMetrics
from hima.experiments.temporal_pooling.stp.temporal_memory import TemporalMemory


class TmTracker:
    tm: TemporalMemory

    def __init__(self, tm: TemporalMemory):
        self.tm = tm

    def on_activate(self, _, reset: bool) -> TMetrics:
        if reset:
            return {}

        tm = self.tm
        return {
            'column/prediction_volume': tm.column_prediction_volume,
            'column/miss_rate': tm.column_miss_rate,
            'column/imprecision': tm.column_imprecision,
            'cell/prediction_volume': tm.cell_prediction_volume,
            'cell/imprecision': tm.cell_imprecision,
        }


def get_tm_tracker(on: dict) -> TmTracker:
    tracked_stream = on['activate']
    tm = tracked_stream.owner.tm
    return TmTracker(tm=tm)
