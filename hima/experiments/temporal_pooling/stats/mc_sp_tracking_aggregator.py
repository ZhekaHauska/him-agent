#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from typing import Any

from hima.common.utils import prepend_dict_keys
from hima.experiments.temporal_pooling.stats.metrics import TMetrics


class SpTrackingCompartmentalAggregator:
    """
    Wrapper class that tracks multiple compartments separately and aggregates their metrics.

    It is tracker agnostic — it dynamically exposes the same interface as the underlying trackers
    and just forwards the calls to them aggregating the results.
    """
    sp: Any
    # dict of compartment_name -> tracker
    compartments: dict[str, Any]
    supported: bool

    def __init__(self, sp, tracker_class, **config):
        self.sp = sp

        self.compartments = {}
        for comp_name, compartment in sp.compartments.items():
            tracker = tracker_class(compartment, **config)
            if tracker.supported:
                self.compartments[comp_name] = tracker

        self.supported = len(self.compartments) > 0
        if not self.supported:
            return

        # extract public API from the tracker class
        self.public_api = {
            method_name: getattr(tracker_class, method_name)
            for method_name in dir(tracker_class)
            if method_name.startswith('on_') and callable(getattr(tracker_class, method_name))
        }

    def __getattr__(self, method_name):
        # dynamically create a method that calls the corresponding method of the
        # underlying trackers and return aggregated results
        cls_method = self.public_api[method_name]

        def _handle_call(*args, **kwargs):
            return self.aggregate_over_compartments({
                comp_name: cls_method(comp_tracker, *args, **kwargs)
                for comp_name, comp_tracker in self.compartments.items()
            })

        # cache the method for future calls => getattr will not be called again for it
        setattr(self, method_name, _handle_call)

        # return method
        return _handle_call

    @staticmethod
    def aggregate_over_compartments(metrics: dict[str, TMetrics]) -> TMetrics:
        result = {}
        for comp_name, comp_metrics in metrics.items():
            if not comp_metrics:
                continue
            result |= prepend_dict_keys(comp_metrics, prefix=comp_name, separator='/')
        return result
