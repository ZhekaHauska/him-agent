#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from hima.modules.belief.utils import normalize
import numpy as np


class Layer:
    """
        This class represents a layer of the neocortex model.
    """
    input_sdr_size: int
    context_input_size: int
    external_input_size: int
    n_obs_vars: int
    n_obs_states: int
    n_hidden_vars: int
    n_hidden_states: int
    n_context_vars: int
    n_context_states: int
    n_external_vars: int
    n_external_states: int
    prediction_columns: np.ndarray | None
    prediction_cells: np.ndarray | None
    observation_messages: np.ndarray
    internal_messages: np.ndarray
    context_messages: np.ndarray
    external_messages: np.ndarray

    def set_external_messages(self, messages=None):
        # update external cells
        if messages is not None:
            self.external_messages = messages
        elif self.external_input_size != 0:
            self.external_messages = normalize(
                np.zeros(self.external_input_size).reshape((self.n_external_vars, -1))
            ).flatten()

    def set_context_messages(self, messages=None):
        # update external cells
        if messages is not None:
            self.context_messages = messages
        elif self.context_input_size != 0:
            self.context_messages = normalize(
                np.zeros(self.context_input_size).reshape((self.n_context_vars, -1))
            ).flatten()

    def make_state_snapshot(self):
        return (
            # mutable attributes:
            self.internal_messages.copy(),
            # immutable attributes:
            self.external_messages,
            self.context_messages,
            self.prediction_cells,
            self.prediction_columns
        )

    def restore_last_snapshot(self, snapshot):
        if snapshot is None:
            return

        (
            self.internal_messages,
            self.external_messages,
            self.context_messages,
            self.prediction_cells,
            self.prediction_columns
        ) = snapshot

        # explicitly copy mutable attributes:
        self.internal_messages = self.internal_messages.copy()

    def reset(self):
        raise NotImplemented

    def predict(self, **_):
        raise NotImplemented

    def observe(
            self,
            observation: np.ndarray,
            learn: bool = True
    ):
        raise NotImplemented
