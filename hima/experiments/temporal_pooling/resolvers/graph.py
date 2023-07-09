#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from abc import ABC

from hima.common.config.base import TConfig
from hima.experiments.temporal_pooling.graph.block import Block


class BlockResolver(ABC):
    family: str = "base"

    @staticmethod
    def resolve(
        global_config: TConfig, config: TConfig,
        block_id: int, block_name: str,
        **kwargs
    ) -> Block:
        raise NotImplementedError()


