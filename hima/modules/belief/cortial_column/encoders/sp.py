#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.modules.belief.cortial_column.encoders.base import BaseEncoder
from hima.experiments.temporal_pooling.stp.sp_grouped import SpatialPoolerGrouped
import numpy as np


class SpatialPooler(BaseEncoder):
    def __init__(self, conf):
        self.model = SpatialPoolerGrouped(**conf)
        self.n_vars = self.model.n_groups
        self.n_states = self.model.group_size

    def encode(self, input_: np.ndarray, learn: bool) -> np.ndarray:
        return self.model.compute(input_, learn)
