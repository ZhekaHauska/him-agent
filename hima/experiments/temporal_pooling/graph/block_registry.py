#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config.global_config import GlobalConfig
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.stream import StreamRegistry


class BlockRegistry:
    blocks_config_key = 'blocks'

    _id: int
    _blocks: dict[str, Block]
    _streams: StreamRegistry
    _global_config: GlobalConfig

    def __init__(self, global_config: GlobalConfig, streams: StreamRegistry):
        self._global_config = global_config
        self._id = 0
        self._blocks = {}
        self._streams = streams

    def __getitem__(self, block_name: str) -> Block:
        block = self._blocks.get(block_name, None)
        if block is None:
            # print(f"Resolving block {block_name}")
            block = self._resolve_block(block_name)
            self._blocks[block_name] = block
        return block

    def _resolve_block(self, block_name: str):
        block_id = self._id
        self._id += 1

        # construct fully specified path
        path = f'{self.blocks_config_key}.{block_name}'
        # collect config and extend it with base block attributes: id and name
        block_config = self._global_config.config_resolver.resolve(
            path,
            config_type=dict
        ) | dict(
            id=block_id, name=block_name, stream_registry=self._streams
        )
        return self._global_config.resolve_object(block_config)

    def __iter__(self):
        yield from self._blocks

    def __contains__(self, item):
        return item in self._blocks
