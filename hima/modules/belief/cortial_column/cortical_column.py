#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional

import numpy as np
from hima.common.metrics import get_surprise
from hima.modules.belief.cortial_column.layer import Layer
from hima.modules.belief.cortial_column.encoders.base import BaseEncoder


class CorticalColumn:
    """
        Class for probabilistic modelling of a cortical column of the neocortex.
        Cortical column consists of several layers and orchestrates them.
    """
    def __init__(
            self,
            layer: Layer,
            encoder: Optional[BaseEncoder],
    ):
        self.layer = layer
        self.encoder = encoder

        self.learn_layer = True
        self.learn_encoder = True
        self.learn_decoder = True

        self.predicted_image = None
        self.encoded_sdr = None
        self.surprise = 0

    def observe(self, local_input, external_input, reward, learn=True):
        # predict current local input step
        if external_input is not None and (self.layer.external_input_size > 0):
            external_messages = np.zeros(self.layer.external_input_size)
            if external_input >= 0:
                external_messages[external_input] = 1
            else:
                external_messages = np.empty(0)
        else:
            external_messages = None

        self.layer.set_external_messages(external_messages)
        self.layer.predict(learn=learn and self.learn_layer)

        self.predicted_image = self.encoder.decode(
            self.layer.prediction_columns,
            learn=learn and self.learn_decoder,
            correct=local_input
        )

        if self.encoder is not None:
            self.encoded_sdr = self.encoder.encode(local_input, learn and self.learn_encoder)
        else:
            self.encoded_sdr = local_input

        self.layer.observe(self.encoded_sdr, reward, learn=learn and self.learn_layer)
        self.layer.set_context_messages(self.layer.internal_messages)

        self.surprise = 0
        if len(self.encoded_sdr) > 0:
            self.surprise = get_surprise(
                self.layer.prediction_columns, self.encoded_sdr, mode='categorical'
            )

    def predict(self, context_messages, external_messages=None):
        self.layer.set_context_messages(context_messages)
        self.layer.set_external_messages(external_messages)
        self.layer.predict()
        self.predicted_image = self.encoder.decode(self.layer.prediction_columns, learn=False)

    def reset(self, context_messages=None, external_messages=None):
        self.layer.reset()
        if context_messages is not None:
            self.layer.set_context_messages(context_messages)
        if external_messages is not None:
            self.layer.set_external_messages(external_messages)

        self.predicted_image = None
        self.encoded_sdr = None
        self.surprise = 0

    def make_state_snapshot(self):
        return (
            # mutable attributes:
            # immutable attributes:
            self.layer.make_state_snapshot(),
            self.predicted_image
        )

    def restore_last_snapshot(self, snapshot):
        if snapshot is None:
            return

        (
            layer_snapshot,
            self.predicted_image
        ) = snapshot

        self.layer.restore_last_snapshot(layer_snapshot)
