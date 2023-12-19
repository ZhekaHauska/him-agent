#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from hima.common.config.base import TConfig
from hima.experiments.temporal_pooling.graph.stream import Stream, SdrStream
from hima.experiments.temporal_pooling.stats.metrics import TMetrics

THandler = Callable[[Stream, Any, bool], None]

if TYPE_CHECKING:
    from hima.experiments.temporal_pooling.graph.model import Model


TRACKING_ENABLED = 'tracking_enabled'


class TrackerBlock:
    model: Model
    name: str
    # stream_name -> tracker.on_xxx
    track: dict
    # each tracker listen special `enabled` stream, which controls trackers' activity
    enabled: Stream

    # tracker object
    tracker: Any

    def __init__(self, model: Model, name: str, tracker: TConfig, on: dict[str, Stream]):
        self.model = model
        self.name = name
        self.tracker = self.model.config.resolve_object(tracker, on=on, model=self.model)
        self.enabled = model.streams[TRACKING_ENABLED]

        self.track = {}
        for handler_name, stream in on.items():
            # register general handler
            stream.track(tracker=self.handle)
            # save particular handler
            handler = self.get_handler(self.tracker, handler_name)
            self.track[stream.name] = handler

    def handle(self, stream: Stream | SdrStream, new_value, reset: bool):
        if not self.enabled.get():
            return

        metrics = self.track[stream.name](new_value, reset=reset)
        metrics = self.personalize_metrics(metrics)
        if metrics:
            self.model.metrics |= metrics

    def personalize_metrics(self, metrics: TMetrics):
        return {
            f'{self.name}/{k}': metrics[k]
            for k in metrics
        }

    @staticmethod
    def get_handler(tracker, handler_name: str) -> THandler:
        return getattr(tracker, f'on_{handler_name}')
