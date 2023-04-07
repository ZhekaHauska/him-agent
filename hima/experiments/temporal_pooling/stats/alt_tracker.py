#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any, Callable, TYPE_CHECKING

from hima.experiments.temporal_pooling.graph.stream import Stream
from hima.experiments.temporal_pooling.graph.model import Model

THandler = Callable[[Stream, Any, bool], None]

if TYPE_CHECKING:
    from hima.experiments.temporal_pooling.graph.model import Model


TMetrics = dict[str, Any]


class AltTracker:
    model: Model
    name: str
    track: dict

    valid: bool

    def __init__(self, model: Model, name: str, on: dict):
        self.model = model
        self.name = name

        self.track = {}
        self.valid = True
        new_streams = []
        for handler_name, stream_name in on:
            handler: THandler = getattr(self, f'on_{handler_name}_updated')
            if stream_name not in self.model:
                new_streams.append(stream_name)
            stream = self.model.try_track_stream(stream_name, handler)
            if stream is None:
                self.valid = False
                break
            self.track[stream.name] = (stream, handler)

        if not self.valid:
            # remove all streams added before
            for stream_name in new_streams:
                self.model.streams.pop(stream_name)

