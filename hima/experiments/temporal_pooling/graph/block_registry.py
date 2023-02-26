#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Any

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.graph.block import Block


class BlockRegistry:
    _global_config: GlobalConfig
    _supplementary_config: dict[str, Any]

    _id: int
    _block_configs: dict[str, TConfig]
    _blocks: dict[str, Block]

    def __init__(
            self, global_config: GlobalConfig, block_configs: TConfig,
            input_sds: Sds,
            **supplementary_config
    ):
        self._global_config = global_config
        self._supplementary_config = supplementary_config
        self.input_sds = input_sds

        self._id = 0
        self._block_configs = block_configs
        self._blocks = {}

    def __getitem__(self, block_name: str) -> Block:
        if block_name not in self._blocks:
            print(f"Resolving block {block_name}")
            self._blocks[block_name] = self._resolve_block(block_name=block_name)
        return self._blocks[block_name]

    def _resolve_block(self, block_name: str):
        block_id = self._id
        self._id += 1

        block_config = self._block_configs[block_name] | dict(
            id=block_id, name=block_name,
        ) | self._supplementary_config

        if block_name == '___':
            block_config |= dict(input_sds=self.input_sds)

        return self._global_config.resolve_object(block_config)

    def build(self) -> dict[str, Block]:
        for name in self._blocks:
            self._blocks[name].build()
        return self._blocks
