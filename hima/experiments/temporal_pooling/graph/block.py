#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from abc import ABC, abstractmethod

from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling.graph.stream import Stream


class Block(ABC):
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
        self._parse_streams(kwargs)

    def register_stream(self, name: str) -> Stream:
        if name not in self.streams:
            self.streams[name] = Stream(name=name, block=self)
        return self.streams[name]

    # --------------- Overrideable public interface ---------------

    # noinspection PyMethodMayBeStatic
    def align_dimensions(self) -> bool:
        """
        Align or induce block's streams dimensions.
        By default, does nothing. Override if it's applicable.
        """
        return True

    def reset(self, **kwargs):
        for name in self.streams:
            self.streams[name].sdr = []

    @abstractmethod
    def build(self, **kwargs):
        """Build block after all its configurable parameters are resolved."""
        raise NotImplementedError()

    @abstractmethod
    def compute(self, data: dict[str, SparseSdr], **kwargs):
        """Make a computation given the provided input data streams."""
        raise NotImplementedError()

    # --------------- String representation ---------------

    @property
    def shortname(self):
        return f'{self.id}_{self.family}'

    @property
    def fullname(self):
        return f'{self.shortname} {self.name}'

    def __repr__(self):
        return self.fullname

    def _parse_streams(self, kwargs: dict):
        for key, value in kwargs.items():
            if not str.endswith(key, '_sds'):
                continue

            stream_name, sds = key[:-4], value
            stream = self.register_stream(stream_name)
            stream.join_sds(sds)
