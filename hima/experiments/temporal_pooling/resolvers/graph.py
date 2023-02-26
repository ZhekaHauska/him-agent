#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from abc import ABC
from typing import Optional, Any

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.config.values import resolve_value
from hima.common.sds import Sds
from hima.common.utils import isnone
from hima.experiments.temporal_pooling.graph.graph import (
    Pipeline,
    ComputationUnit
)
from hima.experiments.temporal_pooling.graph.pipe import Pipe
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.stream import Stream


class BlockResolver(ABC):
    family: str = "base"

    @staticmethod
    def resolve(
        global_config: TConfig, config: TConfig,
        block_id: int, block_name: str,
        **kwargs
    ) -> Block:
        raise NotImplementedError()


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


class PipelineResolver:
    block_registry: BlockRegistry

    _previous_block: Optional[str]
    _current_block: Optional[str]

    def __init__(self, block_registry: BlockRegistry):
        self.block_registry = block_registry
        self._current_block = None
        self._previous_block = None

    def resolve(self, pipeline: list) -> Pipeline:
        # defined with config
        units = []
        self._current_block = self._previous_block = None
        for unit in pipeline:
            print(f'unit: {unit}')
            unit = self.parse_unit(unit)
            units.append(unit)
            self._previous_block = unit.block.name
            self._current_block = None

        # finish building
        self.resolve_dimensions(units)
        blocks = self.block_registry.build()

        return Pipeline(units=units, blocks=blocks)

    @staticmethod
    def resolve_dimensions(units: list[ComputationUnit], max_iters: int = 100):
        unresolved = []
        for i in range(max_iters):
            unresolved = [
                pipe
                for unit in units
                for pipe in unit.connections
                if not pipe.align_dimensions()
            ]
            if not unresolved:
                break

        assert not unresolved, f'Cannot resolve {unresolved} pipeline units!'

    def parse_unit(self, unit) -> ComputationUnit:
        def _parse_unit(
                pipe: str = None, *, sds: Sds = None,
                block: str = None, pipes: list[str, dict] = None
        ) -> ComputationUnit:
            if block is not None:
                # it is used to resolve pipes during parsing
                self._current_block = block

            pipes = isnone(pipes, [pipe])
            connections = []
            for pipe in pipes:
                if isinstance(pipe, dict):
                    pipe = self.parse_pipe(**pipe)
                else:
                    pipe = self.parse_pipe(pipe, sds=sds)
                connections.append(pipe)

            return ComputationUnit(connections=connections)

        if isinstance(unit, dict):
            return _parse_unit(**unit)
        elif isinstance(unit, str):
            return _parse_unit(pipe=unit)

    def parse_pipe(self, pipe: str, *, sds: Any = None) -> Pipe:
        src, dst = pipe.strip().split('->')
        src = self.parse_stream(src, default_block=self._previous_block)
        dst = self.parse_stream(dst, default_block=self._current_block)
        pipe = Pipe(src=src, dst=dst, sds=sds)
        return pipe

    def parse_stream(self, s: str, default_block: str = None) -> Stream:
        stream_parts = s.strip().split('.')
        if len(stream_parts) == 1:
            # default block with the given stream name
            block_name = default_block
            stream_name = stream_parts[0]
        elif len(stream_parts) == 2:
            block_name, stream_name = stream_parts
            # resolve with default if needed
            block_name = resolve_value(block_name, substitute_with=default_block)
        else:
            raise ValueError(f'Cannot parse stream from "{s}"')

        stream = self.block_registry[block_name].register_stream(stream_name)
        return stream
