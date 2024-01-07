#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Union, Optional

import numpy as np
from htm.bindings.sdr import SDR

from hima.common.metrics import get_surprise
from hima.modules.baselines.hmm import FCHMMLayer
from hima.modules.baselines.lstm import LstmLayer
from hima.modules.baselines.rwkv import RwkvLayer
from hima.modules.belief.cortial_column.layer import Layer
from hima.modules.htm.spatial_pooler import SPEnsemble, SPDecoder


class CorticalColumn:
    """
        Class for probabilistic modelling of a cortical column of the neocortex.
        Cortical column consists of several layers and orchestrates them.
    """
    def __init__(
            self,
            layer: Union[Layer, FCHMMLayer, LstmLayer, RwkvLayer],
            encoder: Optional[SPEnsemble],
            decoder: Optional[SPDecoder]
    ):
        self.layer = layer
        self.encoder = encoder
        self.decoder = decoder

        self.predicted_image = None
        self.surprise = 0

        if self.encoder is not None:
            self.input_sdr = SDR(self.encoder.getNumInputs())
            self.output_sdr = SDR(self.encoder.getNumColumns())
        else:
            self.input_sdr = SDR(self.layer.input_sdr_size)
            self.output_sdr = SDR(self.layer.input_sdr_size)

    def observe(self, local_input, external_input, learn=True):
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
        self.layer.predict(learn=learn)

        self.input_sdr.sparse = local_input

        if self.decoder is not None:
            self.predicted_image = self.decoder.decode(
                self.layer.prediction_columns, learn=learn, correct_obs=self.input_sdr.dense
            )
        else:
            self.predicted_image = self.layer.prediction_columns

        # observe real outcome and optionally learn using prediction error
        if self.encoder is not None:
            self.encoder.compute(self.input_sdr, learn, self.output_sdr)
        else:
            self.output_sdr.sparse = self.input_sdr.sparse

        self.layer.observe(self.output_sdr.sparse, learn=learn)
        self.layer.set_context_messages(self.layer.internal_forward_messages)

        self.surprise = 0
        encoded_obs = self.output_sdr.sparse
        if len(encoded_obs) > 0:
            self.surprise = get_surprise(
                self.layer.prediction_columns, encoded_obs, mode='categorical'
            )

    def predict(self, context_messages, external_messages=None):
        self.layer.set_context_messages(context_messages)
        self.layer.set_external_messages(external_messages)
        self.layer.predict()

        if self.decoder is not None:
            self.predicted_image = self.decoder.decode(self.layer.prediction_columns, learn=False)
        else:
            self.predicted_image = self.layer.prediction_columns

    def reset(self, context_messages, external_messages):
        self.layer.reset()
        self.layer.set_context_messages(context_messages)
        self.layer.set_external_messages(external_messages)

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
