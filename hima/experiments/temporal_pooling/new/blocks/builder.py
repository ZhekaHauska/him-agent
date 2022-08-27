#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from abc import ABC
from typing import Optional, Any

from hima.common.config_utils import (
    extracted_family, resolve_nested_configs, TConfig, is_resolved_value, resolve_value
)
from hima.common.sds import Sds
from hima.common.utils import isnone
from hima.experiments.temporal_pooling.new.blocks.block_resolver import BlockResolver
from hima.experiments.temporal_pooling.new.blocks.computational_graph import (
    Pipe, Block, Pipeline,
    ComputationUnit, Stream, ExternalApiBlock
)
from hima.experiments.temporal_pooling.new.blocks.dataset_resolver import (
    DataGeneratorResolver
)


class BlockRegistryResolver:
    config: TConfig
    supplementary_config: dict[str, Any]

    blocks: dict[str, Block]

    _block_configs: dict[str, TConfig]
    _block_resolvers: dict[str, BlockResolver]
    _id: int

    def __init__(self, config: TConfig, block_configs: TConfig, **supplementary_config):
        self.config = config
        self.supplementary_config = supplementary_config
        self.blocks = {}

        self._id = 0
        self._block_configs = block_configs
        self._block_resolvers = {
            DataGeneratorResolver.family: DataGeneratorResolver()
        }

    def __getitem__(self, item: str) -> Block:
        if item not in self.blocks:
            self.blocks[item] = self._resolve_block(block_name=item)
        return self.blocks[item]

    def _resolve_block(self, block_name: str):
        block_id = self._id
        self._id += 1

        if block_name == '___':
            return ExternalApiBlock(id=block_id, name=block_name)

        block_config, block_family = extracted_family(config=self._block_configs[block_name])
        block_config = resolve_nested_configs(
            config_registry=self.config[block_family], config=block_config
        )
        return self._block_resolvers[block_family].resolve(
            global_config=self.config,
            config=block_config,
            block_id=block_id,
            block_name=block_name,
            **self.supplementary_config
        )


class PipelineResolver:
    block_registry: BlockRegistryResolver

    _previous_block: Optional[str]
    _current_block: Optional[str]

    def __init__(self, block_registry: BlockRegistryResolver):
        self.block_registry = block_registry
        self._current_block = None
        self._previous_block = None

    def resolve(self, pipeline: list) -> Pipeline:
        # defined with config
        units = []
        self._current_block = self._previous_block = None
        for unit in pipeline:
            unit = self.parse_unit(unit)
            units.append(unit)
            self._previous_block = unit.block.name
            self._current_block = None

        return Pipeline(units, block_registry=self.block_registry.blocks)

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

        block = self.block_registry[block_name]
        return Stream(name=stream_name, block=block)


def _resolve_interface(streams: list[str], default_streams: list[str]) -> list[str]:
    # None means default, unresolved means do nothing it is inducted later
    streams = resolve_value(streams)
    streams = isnone(streams, default_streams)

    return streams if is_resolved_value(streams) else []
