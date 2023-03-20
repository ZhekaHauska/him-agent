#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from abc import abstractmethod

from hima.experiments.temporal_pooling.graph.node import Node
from hima.experiments.temporal_pooling.graph.stream import SdrStream, StreamRegistry


class Block(Node):
    """Base building block of the computational graph / neural network."""

    family: str = 'base_block'
    supported_streams: set[str] = {}

    id: int
    name: str
    stream_registry: StreamRegistry

    # TODO:
    #  1. log to charts, what to log?
    #  2. rename tag to ? and consider removing id

    def __init__(self, id: int, name: str, stream_registry: StreamRegistry, **kwargs):
        self.id = id
        self.name = name
        self.stream_registry = stream_registry
        self._config = self._extract_streams(kwargs)

    # ------------ Block overrideable public interface ------------
    @abstractmethod
    def compile(self):
        """Build block after all its configurable parameters are resolved."""
        raise NotImplementedError()

    def reset(self, **kwargs):
        for name in self.stream_registry:
            self.stream_registry[name].write([])

    # ----------------- Node public interface ---------------------
    def expand(self):
        yield self

    # noinspection PyMethodMayBeStatic
    def align_dimensions(self) -> bool:
        """
        Align or induce block's streams dimensions.
        By default, does nothing. Override if it's applicable.
        """
        return True

    def forward(self) -> None:
        """Blocks are supposed to have conscious `forward` methods that are used via BlockCall."""
        raise ValueError(
            'Blocks are supposed to have conscious `forward` methods that are used via BlockCall.'
        )

    def __repr__(self) -> str:
        return self.fullname

    # ------------------ String representation --------------------
    @property
    def shortname(self):
        return f'{self.id}_{self.family}'

    @property
    def fullname(self):
        return f'{self.shortname} {self.name}'

    # ---------------------- Utility methods ----------------------
    def stream_name(self, short_name):
        qualified_name = f'{self.name}.{short_name}'
        return qualified_name

    def __getitem__(self, stream_name):
        return self.stream_registry[self.stream_name(stream_name)]

    def _extract_streams(self, kwargs: dict) -> dict:
        config = {}
        for key, value in kwargs.items():
            if not str.endswith(key, '_sds'):
                config[key] = value
                continue

            stream_name, sds = f'{key[:-4]}.sdr', value
            stream_name = self.stream_name(stream_name)

            stream: SdrStream = self.stream_registry.register(stream_name, owner=self)
            stream.set_sds(sds)
        return config
