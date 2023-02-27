#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from abc import abstractmethod

from hima.experiments.temporal_pooling.graph.node import Node
from hima.experiments.temporal_pooling.graph.stream import Stream


class Block(Node):
    """Base building block of the computational graph / neural network."""

    family: str = "base_block"
    supported_streams: set[str] = {}

    id: int
    name: str
    streams: dict[str, Stream]

    # TODO:
    #  1. log to charts, what to log?
    #  2. rename tag to ? and consider removing id

    def __init__(self, id: int, name: str, **kwargs):
        self.id = id
        self.name = name
        self.streams = {}
        self._config = self._extract_streams(kwargs)

    # ---------- Block non-overrideable public interface ----------

    def register_stream(self, name: str) -> Stream:
        if name not in self.streams:
            self.streams[name] = Stream(name=name, block=self)
        return self.streams[name]

    # ------------ Block overrideable public interface ------------

    def reset(self, **kwargs):
        for name in self.streams:
            self.streams[name].sdr = []

    @abstractmethod
    def compile(self):
        """Build block after all its configurable parameters are resolved."""
        raise NotImplementedError()

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
    def _extract_streams(self, kwargs: dict):
        config = {}
        for key, value in kwargs.items():
            if not str.endswith(key, '_sds'):
                config[key] = value
                continue

            stream_name, sds = key[:-4], value
            stream = self.register_stream(stream_name)
            stream.join_sds(sds)
        return config
