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
from hima.experiments.temporal_pooling.graph.block_call import BlockCall
from hima.experiments.temporal_pooling.graph.block_registry import BlockRegistry
from hima.experiments.temporal_pooling.graph.model import Model
from hima.experiments.temporal_pooling.graph.node import Node
from hima.experiments.temporal_pooling.graph.pipe import Pipe
from hima.experiments.temporal_pooling.graph.pipeline import Pipeline
from hima.experiments.temporal_pooling.graph.repeat import Repeat
from hima.experiments.temporal_pooling.graph.stream import Stream, StreamRegistry, SdrStream


class ModelCompiler:
    blocks: BlockRegistry
    streams: StreamRegistry

    def __init__(self, global_config: GlobalConfig):
        self.streams = StreamRegistry()
        self.blocks = BlockRegistry(global_config, self.streams)

    def parse(self, pipeline: list) -> Model:
        return Model(
            pipeline=self.parse_pipeline(pipeline=pipeline),
            blocks=self.blocks,
            streams=self.streams
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

    def parse_pipeline(self, **kwargs) -> Pipeline:
        pipeline_name, pipeline = Pipeline.extract_args(**kwargs)
        return Pipeline(
            name=pipeline_name,
            pipeline=[self.parse_node(unit) for unit in pipeline]
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
            elif len(node) == 1:
                return self.parse_pipeline(**node)

        raise ValueError(f'Cannot interpret a node from {node}.')

    def parse_repeat(self, repeat: int, do: TConfig) -> Repeat:
        return Repeat(repeat=repeat, do=self.parse_pipeline(do=do))

    def parse_block_func_call(self, call: str) -> BlockCall:
        # pattern: block.func
        block_name, func_name = call.strip().split('.')
        block = self.blocks[block_name]
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
        src = self.parse_stream(src)
        dst = self.parse_stream(dst)
        pipe = Pipe(src=src, dst=dst, sds=sds)

        return pipe

    def parse_stream(self, stream_name: str) -> Stream | SdrStream:
        # patterns: block.stream.sdr | block.stream | stream.sdr | stream
        stream_name = stream_name.strip()
        is_sdr = stream_name.endswith('.sdr')

        name_parts = stream_name.split('.')
        n_name_parts = len(name_parts) - is_sdr

        if n_name_parts == 2:
            # register block
            block = self.blocks[name_parts[0]]
            return self.streams.register(stream_name, owner=block)
        else:
            return self.streams.register(stream_name)
