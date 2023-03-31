#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from hima.experiments.temporal_pooling.graph.node import Stretchable
from hima.experiments.temporal_pooling.graph.stream import SdrStream


if TYPE_CHECKING:
    from hima.experiments.temporal_pooling.graph.model import Model


class Block(Stretchable):
    """Base building block of the computational graph / neural network."""

    family: str = 'base_block'
    name: str

    supported_streams: set[str] = {}
    model: Model

    def __init__(self, name: str, model: Model, **kwargs):
        self.name = name
        self.model = model
        # Blocks MUST register themselves
        self.model.blocks[name] = self
        self._config = self._extract_sdr_streams(kwargs)

    @abstractmethod
    def compile(self):
        """Build block after all its configurable parameters are resolved."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.name

    # ---------------------- Utility methods ----------------------
    def qualified_stream_name(self, stream_short_name: str) -> str:
        return f'{self.name}.{stream_short_name}'

    def __getitem__(self, stream_name):
        return self.model.streams[self.qualified_stream_name(stream_name)]

    def _extract_sdr_streams(self, kwargs: dict) -> dict:
        # the block itself has to be registered before starting register its streams
        assert self.name in self.model.blocks

        config = {}
        for key, value in kwargs.items():
            if not str.endswith(key, '_sds'):
                config[key] = value
                continue

            stream_name, sds = f'{key[:-4]}.sdr', value
            stream_name = self.qualified_stream_name(stream_name)

            stream: SdrStream = self.model.register_stream(stream_name)
            stream.set_sds(sds)
        return config
