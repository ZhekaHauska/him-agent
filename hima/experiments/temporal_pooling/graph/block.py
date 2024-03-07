#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from hima.common.config.base import TConfig
from hima.experiments.temporal_pooling.graph.node import Stretchable, Stateful

if TYPE_CHECKING:
    from hima.experiments.temporal_pooling.graph.model import Model


class Block(Stretchable, Stateful):
    """Base building block of the computational graph / neural network."""

    family: str = 'base_block'
    name: str

    # define in a class as set, then during init it will be modified to dict[short_name] = stream
    supported_streams: set[str] | dict[str, str] = {}
    model: Model

    def __init__(self, name: str, model: Model, **kwargs):
        self.name = name
        self.model = model
        # Blocks MUST register themselves
        self.model.blocks[name] = self

        # extend set to dict with values — resolved stream full names
        self.supported_streams = {
            short_name: self.to_full_stream_name(short_name)
            for short_name in self.supported_streams
        }

        self.extract_sdr_streams(kwargs)

    @abstractmethod
    def compile(self):
        """Build block after all its configurable parameters are resolved."""
        raise NotImplementedError()

    def reset(self):
        """Overload if something needs resetting."""
        for stream_name in self.supported_streams:
            stream = self[stream_name]
            if stream and stream.is_sdr:
                stream.set([], reset=True)

    def __repr__(self) -> str:
        return self.name

    # ---------------------- Utility methods ----------------------
    def __getitem__(self, stream_short_name):
        return self.model.streams.get(self.supported_streams[stream_short_name], None)

    def extract_sdr_streams(self, kwargs: TConfig):
        # the block itself has to be registered before starting register its streams
        assert self.name in self.model

        for key, value in kwargs.items():
            if not key.endswith('_sds'):
                continue

            short_name, sds = f'{key[:-4]}.sdr', value
            self.register_stream(short_name).set_sds(sds)

    def register_stream(self, short_name: str):
        stream_name = self.supported_streams[short_name]
        return self.model.register_stream(stream_name)

    def to_full_stream_name(self, short_name: str) -> str:
        return f'{self.name}.{short_name}'
