#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any, Callable

from hima.experiments.temporal_pooling.graph.model import Model
from hima.experiments.temporal_pooling.graph.stream import Stream

THandler = Callable[[Stream, Any, bool], None]


class AltTracker:
    model: Model
    name: str
    track: dict

    def __init__(self, model: Model, name, on: dict):
        self.model = model
        self.name = name
        self.track = on

        streams = self.model.streams
        for handler_name, stream_name in on:
            handler: THandler = getattr(self, f'on_{name}_updated')
            streams.track(stream_name, handler)
