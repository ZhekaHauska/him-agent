#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from typing import Any

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.config.values import get_unresolved_value
from hima.common.run.argparse import parse_str
from hima.experiments.temporal_pooling.blocks.tracker import TrackerBlock
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.block_call import BlockCall
from hima.experiments.temporal_pooling.graph.node import Node, Stretchable, Stateful
from hima.experiments.temporal_pooling.graph.pipe import Pipe, SdrPipe
from hima.experiments.temporal_pooling.graph.pipeline import Pipeline
from hima.experiments.temporal_pooling.graph.repeat import Repeat
from hima.experiments.temporal_pooling.graph.stream import Stream, SdrStream
from hima.experiments.temporal_pooling.stats.metrics import TMetrics


class Model(Stretchable, Stateful, Node):
    blocks_config_key = 'blocks'

    config: GlobalConfig
    nodes: list[Node]
    pipeline: Pipeline
    streams: dict[str, Stream | SdrStream]
    blocks: dict[str, Block]
    trackers: dict

    metrics: TMetrics

    def __init__(
            self,
            global_config: GlobalConfig,
            pipeline: Pipeline | list,
            external: list[str],
            track: list[TConfig]
    ):
        self.config = global_config
        self.nodes = []
        self.streams = {}
        self.blocks = {}

        if not isinstance(pipeline, Pipeline):
            pipeline = self.config.resolve_object(
                pipeline, object_type_or_factory=self.parse, config_type=list
            )
        self.pipeline = pipeline

        for external_var in external:
            self.register_stream(external_var)

        # delay trackers creation until the rest of the model is compiled as it requires dimensions
        # noinspection PyTypeChecker
        self.trackers = track
        self.metrics = {}

    def compile(self):
        self.fit_dimensions()

        for _, block in self.blocks.items():
            block.compile()

        # create and register trackers
        self.trackers, track = {}, self.trackers
        for tracker in track:
            self.try_register_tracker(**tracker)

    def fit_dimensions(self, max_iters: int = 100) -> bool:
        unaligned_objects = [
            node
            for node in self.nodes
            if isinstance(node, Stretchable)
        ] + [
            block
            for _, block in self.blocks.items()
            if isinstance(block, Stretchable)
        ]

        for i in range(max_iters):
            unaligned_objects = [
                obj
                for obj in unaligned_objects
                if not obj.fit_dimensions()
            ]
            if not unaligned_objects:
                break

        assert not unaligned_objects, f'Cannot align the following objects: {unaligned_objects}'
        return True

    def reset(self):
        # TODO: implement reset policy
        pass

    def forward(self) -> None:
        self.pipeline.forward()

    def __repr__(self) -> str:
        return f'{self.pipeline}'

    def __contains__(self, item: str):
        # allow both just for simpler and shorter usage. We expect blocks and streams have
        # separate contexts such that the usage is unambiguous
        return item in self.blocks or item in self.streams

    def flush_metrics(self) -> TMetrics:
        metrics = self.metrics
        self.metrics = {}
        return metrics

    # =========== Blocks/Streams API =========

    def resolve_block(self, name: str) -> Block:
        """
        Resolves a block with the given name. If it doesn't exist, creates it.
        NB: Blocks are self-registering, i.e. they register themselves.
        The reason behind it is they can register new streams during they initialization,
        which require the block itself to be already registered. Otherwise, block's streams
        registration process will initiate resolving the owning block and will lead to infinite
        loop.
        Therefore, we cannot register blocks AFTER their initialization, but make them register
        themselves during this process before they need to start registering their streams.
        """
        block = self.blocks.get(name, None)
        if block:
            return block

        # print(f"Resolving block {block_name}")

        # construct fully specified path
        path = f'{self.blocks_config_key}.{name}'
        # collect config and extend it with base block attributes: id and name
        block_config = self.config.config_resolver.resolve(
            config=path,
            config_type=dict
        ) | dict(
            name=name, model=self
        )

        return self.config.resolve_object(block_config)

    def register_stream(self, name: str, allow_block_resolve=True) -> Stream | SdrStream | None:
        # sanitize name first
        name = name.strip()

        # check if it's already registered
        stream = self.streams.get(name, None)
        if stream:
            return stream

        # patterns: block.stream.sdr OR block.stream OR stream.sdr OR stream
        name_parts = name.split('.')
        is_sdr = name.endswith('.sdr')
        is_owned_by_block = len(name_parts) - is_sdr == 2

        # get the owning block (and try to register it too)
        block = None
        if is_owned_by_block:
            if not allow_block_resolve:
                return None
            # if the stream is owned by a block, register it too
            block_name = name_parts[0]
            block = self.resolve_block(block_name)

        # construct a stream object, add it to the registry and return
        stream_class = SdrStream if is_sdr else Stream
        stream = stream_class(name, block=block)
        self.streams[stream.name] = stream
        return stream

    def try_register_tracker(self, name: str, tracker: TConfig, on: dict):
        # ensure all streams are valid, i.e. either exist or belong to existing blocks
        valid, non_existed_streams = True, []
        for handler_name, stream_name in on.items():
            stream = self.streams.get(stream_name)
            if stream is None:
                non_existed_streams.append(stream_name)

            stream = self.register_stream(stream_name, allow_block_resolve=False)
            if stream is None:
                valid = False
                break

            on[handler_name] = stream

        if not valid:
            # pop last as it's not valid, i.e. haven't been created
            non_existed_streams.pop()

            # not all stream are valid ==> abort: don't create tracker and remove all new streams
            for stream_name in non_existed_streams:
                self.streams.pop(stream_name)
            print(f'- {name} tracker')
            return

        self.trackers[name] = TrackerBlock(model=self, name=name, tracker=tracker, on=on)
        print(f'+ {name} tracker')

    # =========== Parse API =========

    def parse(self, *pipeline: list) -> Pipeline:
        return self.parse_pipeline(pipeline=pipeline)

    def parse_pipeline(self, **kwargs) -> Pipeline:
        assert len(kwargs) == 1

        (pipeline_name, pipeline), = kwargs.items()
        # NB: all nodes are aggregated via pipelines.
        # Therefore, this is the only place for node registering
        pipeline_nodes = [self.parse_node(unit) for unit in pipeline]
        self.nodes.extend(pipeline_nodes)

        pipeline = Pipeline(name=pipeline_name, pipeline=pipeline_nodes)
        self.nodes.append(pipeline)
        return pipeline

    def parse_node(self, node: str) -> Node:
        # could be:
        #   - pipe forwarding
        #   - basic block computation
        #   - control block: repeat | if
        # print(f'Parse node {node}')
        if isinstance(node, str):
            node = node.strip()
            # pipe or block call
            if '->' in node:
                return self.parse_pipe(node)
            elif node.endswith('()'):
                return self.parse_block_func_call(node)
        elif isinstance(node, dict):
            if 'if' in node:
                # if control block
                raise NotImplementedError('If block is not yet implemented')
            elif 'repeat' in node:
                # repeat control block
                return self.parse_repeat(**node)
            elif len(node) == 1:
                return self.parse_pipeline(**node)

        raise ValueError(f'Cannot interpret a node from {node}.')

    def parse_repeat(self, repeat: int, do: TConfig) -> Repeat:
        return Repeat(repeat=repeat, do=self.parse_pipeline(do=do))

    def parse_block_func_call(self, call: str) -> BlockCall:
        # pattern: block.func()
        block_name, func_name = call[:-2].split('.')
        block = self.resolve_block(block_name)
        return BlockCall(block=block, name=func_name)

    def parse_pipe(self, pipe: str, *, sds: Any = get_unresolved_value()) -> Pipe:
        # pattern: src -> dst | sds
        pipe = pipe.split('|')
        if len(pipe) == 2:
            pipe, sds = pipe
            sds = parse_str(sds.strip())
        else:
            pipe, = pipe

        src, dst = pipe.strip().split('->')
        src = self.register_stream(src)
        dst = self.register_stream(dst)

        if src.is_sdr:
            pipe = SdrPipe(src=src, dst=dst, sds=sds)
        else:
            pipe = Pipe(src=src, dst=dst)
        return pipe
