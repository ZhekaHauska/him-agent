#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Optional, Any

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.config.values import resolve_value, get_unresolved_value
from hima.common.run.argparse import parse_str
from hima.experiments.temporal_pooling.graph.block_registry import BlockRegistry
from hima.experiments.temporal_pooling.graph.block_call import BlockCall
from hima.experiments.temporal_pooling.graph.model import Model
from hima.experiments.temporal_pooling.graph.node import Node
from hima.experiments.temporal_pooling.graph.pipe import Pipe
from hima.experiments.temporal_pooling.graph.pipeline import Pipeline
from hima.experiments.temporal_pooling.graph.repeat import Repeat
from hima.experiments.temporal_pooling.graph.stream import Stream


class ModelCompiler:
    block_registry: BlockRegistry

    _previous_block: Optional[str]

    def __init__(self, global_config: GlobalConfig):
        self.block_registry = BlockRegistry(global_config)

    def parse(self, api_block: str, pipeline: list) -> Model:
        self._previous_block = None

        return Model(
            api_block=api_block,
            pipeline=self.parse_pipeline(pipeline),
            blocks=self.block_registry.blocks,
        )

    def compile(self, model: Model):
        self.align_dimensions(model)

        # compile blocks
        blocks = model.blocks
        for name in blocks:
            blocks[name].compile()

        return model

    @staticmethod
    def align_dimensions(model: Model, max_iters: int = 100):
        unaligned_nodes = list(model.expand())
        for i in range(max_iters):
            unaligned_nodes = [
                node
                for node in unaligned_nodes
                if not node.align_dimensions()
            ]
            if not unaligned_nodes:
                break

        assert not unaligned_nodes, f'Cannot align nodes: {unaligned_nodes}'

    def parse_pipeline(self, pipeline: list) -> Pipeline:
        return Pipeline(
            self.parse_node(unit) for unit in pipeline
        )

    def parse_node(self, node: str | TConfig) -> Node:
        # could be:
        #   - pipe forwarding
        #   - basic block computation
        #   - control block: repeat | if
        # print(f'Parse node {node}')

        if isinstance(node, str):
            # pipe or block call
            if '->' in node:
                return self.parse_pipe(node)
            else:
                return self.parse_block_func_call(node)
        elif isinstance(node, dict):
            if 'pipe' in node:
                return self.parse_pipe(**node)
            elif 'block' in node:
                return self.parse_block_func_call(**node)
            elif 'if' in node:
                # if control block
                ...
            elif 'repeat' in node:
                # repeat control block
                return self.parse_repeat(**node)

        raise ValueError(f'Cannot interpret a node from {node}.')

    def parse_repeat(self, repeat: int, pipeline: TConfig) -> Repeat:
        return Repeat(repeat=repeat, pipeline=self.parse_pipeline(pipeline))

    def parse_block_func_call(self, call: str, *, block: str = None) -> BlockCall:
        # pattern: block.func or func
        # block could be: name or ???
        func_parts = call.strip().split('.')
        if len(func_parts) == 1:
            # default block with the given func name
            block_name = block
            func_name = func_parts[0]
        elif len(func_parts) == 2:
            block_name, func_name = func_parts
            # resolve with default if needed
            block_name = resolve_value(block_name, substitute_with=block)
        else:
            raise ValueError(f'Cannot parse block func call unit from "{call}"')

        block = self.block_registry[block_name]

        # switch previous block with current block
        self._previous_block = block_name
        return BlockCall(block=block, name=func_name)

    def parse_pipe(self, pipe: str, *, sds: Any = get_unresolved_value()) -> Pipe:
        # pattern: src -> dst | sds
        pipe = pipe.split('|')
        if len(pipe) == 2:
            pipe, sds = pipe
            sds = parse_str(sds)
        else:
            pipe = pipe[0]

        src, dst = pipe.strip().split('->')
        # allows only explicit substitution as in ???.stream_name
        src = self.parse_stream(src, substitution_block=self._previous_block)
        # allows explicit substitution with previous block and current src block implicit default
        dst = self.parse_stream(
            dst,
            default_block=src.block.name,
            substitution_block=self._previous_block
        )
        pipe = Pipe(src=src, dst=dst, sds=sds)

        # switch previous block with current destination
        self._previous_block = pipe.dst.block.name
        return pipe

    def parse_stream(
            self, s: str,
            default_block: str = None,
            substitution_block: str = get_unresolved_value()
    ) -> Stream:
        # pattern: block.stream or stream
        # block could be: name or ???
        stream_parts = s.strip().split('.')
        if len(stream_parts) == 1:
            # default block with the given stream name
            block_name = default_block
            stream_name = stream_parts[0]
        elif len(stream_parts) == 2:
            block_name, stream_name = stream_parts
            # resolve with substitution if needed
            block_name = resolve_value(block_name, substitute_with=substitution_block)
        else:
            raise ValueError(f'Cannot parse stream from "{s}"')

        stream = self.block_registry[block_name].register_stream(stream_name)
        return stream
