#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.config.types import TTypeOrFactory, LazyTypeResolver


class StpLazyTypeResolver(LazyTypeResolver):
    def resolve(self, type_tag: str) -> TTypeOrFactory:
        if type_tag == 'tp.layered':
            from hima.experiments.temporal_pooling.test_on_obs_layered import StpExperiment
            return StpExperiment
