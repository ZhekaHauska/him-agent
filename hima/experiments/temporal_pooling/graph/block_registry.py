#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config.global_config import GlobalConfig
from hima.experiments.temporal_pooling.graph.block import Block


class BlockRegistry:
    blocks_config_key = 'blocks'
    global_config: GlobalConfig

    id: int
    blocks: dict[str, Block]

    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.id = 0
        self.blocks = {}

    def __getitem__(self, block_name: str) -> Block:
        if block_name not in self.blocks:
            # print(f"Resolving block {block_name}")
            self.blocks[block_name] = self._resolve_block(block_name)
        return self.blocks[block_name]

    def _resolve_block(self, block_name: str):
        block_id = self.id
        self.id += 1

        # construct fully specified path
        path = f'{self.blocks_config_key}.{block_name}'
        # collect config and extend it with base block attributes: id and name
        block_config = self.global_config.config_resolver.resolve(
            path,
            config_type=dict
        ) | dict(
            id=block_id, name=block_name,
        )
        return self.global_config.resolve_object(block_config)
