#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from htm.bindings.sdr import SDR
from hima.common.sdr import sparse_to_dense
from hima.modules.belief.utils import softmax, normalize, sample_categorical_variables

import numpy as np
from typing import Literal
from scipy.special import rel_entr
from functools import partial

EPS = 1e-24
INT_TYPE = "int64"
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
PRIOR_MODE = Literal['uniform', 'dirichlet', 'chinese-restaurant']


class TraceBasedLoc:
    def __init__(
            self,
            n_obs_states: int,
            cells_per_column: int,
            column_prior_mode: PRIOR_MODE = "dirichlet",
            alpha: float = 1.0,
            gamma: float = 0.9,
            otp_lr: float = 0.01,
            otp_beta: float = 1.0,
            freq_lr: float = 0.1,
            new_state_weight: float = 1.0,
            seed: int = None,
    ):
        self._rng = np.random.default_rng(seed)

        self.learn = True
        self.cells_per_column = cells_per_column
        self.n_hidden_states = cells_per_column * n_obs_states
        self.n_obs_states = n_obs_states

        self.gamma = gamma
        self.alpha = alpha
        self.otp_weights = self._rng.dirichlet(
            alpha=[self.alpha] * self.n_obs_states,
            size=self.n_hidden_states
        )
        self.otp_lr = otp_lr
        self.otp_beta = otp_beta

        # online Chinese Restaurant Prior
        self.column_prior_mode = column_prior_mode
        self.state_activation_freq = np.zeros(self.n_hidden_states)
        self.freq_lr = freq_lr
        self.new_state_weight = new_state_weight

        self.internal_messages = np.zeros(
            self.n_hidden_states,
            dtype=REAL64_DTYPE
        )
        self.observation_messages = np.zeros(self.n_obs_states)
        self.observation_trace = np.zeros(self.n_obs_states)
        self.norm_observation_trace = np.zeros(self.n_obs_states)

        self.internal_active_state = SDR(self.n_hidden_states)

    def reset(self):
        self.internal_messages = np.zeros(
            self.n_hidden_states,
            dtype=REAL64_DTYPE
        )
        self.observation_messages = np.zeros(self.n_obs_states)
        self.observation_trace = np.zeros(self.n_obs_states)
        self.internal_active_state.sparse = []

    def observe(self, state):
        observation_messages = sparse_to_dense(
            state,
            size=self.n_obs_states,
            dtype=REAL64_DTYPE
        )
        self.observation_trace = observation_messages + self.gamma * self.observation_trace
        self.norm_observation_trace = normalize(
            self.observation_trace[None, :]
        )

        # update messages
        self.observation_messages = observation_messages
        self.internal_messages = self._get_posterior()

        if self.learn:
            self.internal_active_state.sparse = self._sample_cells(self.internal_messages[None, :])
            self._update_frequencies()
            self._update_weights()

    def _get_posterior(self):
        observation = np.flatnonzero(self.observation_messages)
        cells = self._get_cells_for_observation(observation)
        obs_factor = sparse_to_dense(cells, like=self.internal_messages)

        messages = obs_factor

        if self.column_prior_mode == "dirichlet":
            column_prior = self._rng.dirichlet(
                alpha=[self.alpha] * self.cells_per_column
            ).flatten()
            prior = np.zeros_like(obs_factor)
            prior[obs_factor == 1] = column_prior
        elif self.column_prior_mode == "uniform":
            prior = obs_factor
        elif self.column_prior_mode == 'chinese-restaurant':
            eps = self.new_state_weight / (1 / (self.freq_lr + EPS) - 1 + EPS)
            column_prior = self.state_activation_freq[cells].reshape(-1, self.cells_per_column)
            zero_mask = np.isclose(column_prior, 0)
            free_cells_count = np.count_nonzero(
                zero_mask,
                axis=-1
            )
            prob_per_cell = np.divide(
                eps / (1 + eps), free_cells_count,
                where=free_cells_count != 0
            )
            prob_per_cell = np.repeat(prob_per_cell, self.cells_per_column).reshape(
                -1, self.cells_per_column
            )

            column_prior[zero_mask] = prob_per_cell[zero_mask]
            column_prior[~zero_mask] /= (1 + eps)

            prior = np.zeros_like(obs_factor)
            prior[obs_factor == 1] = column_prior.flatten()
        else:
            raise ValueError(f"There is no such column prior mode: {self.column_prior_mode}!")

        dkl = np.sum(rel_entr(self.norm_observation_trace, self.otp_weights[cells]), axis=-1)
        column_probs = np.apply_along_axis(
            partial(softmax, beta=self.otp_beta),
            axis=-1,
            arr=dkl.reshape(
                -1, self.cells_per_column
            )
        )
        ot_prior = np.zeros_like(obs_factor)
        ot_prior[obs_factor == 1] = column_probs.flatten()
        prior *= ot_prior

        prior = prior.reshape(1, -1)
        messages, self.n_bursting_vars = normalize(
            messages * prior, prior, return_zeroed_variables_count=True
        )

        return messages.flatten()

    def _update_frequencies(self, mask=None):
        freq_lr = np.full_like(self.state_activation_freq, fill_value=self.freq_lr)
        freq_lr[np.isclose(self.state_activation_freq, 0)] = 1.0

        delta = self.freq_lr * (self.internal_active_state.dense - self.state_activation_freq)
        if mask is not None:
            self.state_activation_freq[mask] += delta[mask]
        else:
            self.state_activation_freq += delta

        np.clip(self.state_activation_freq, 0, 1, out=self.state_activation_freq)

    def _update_weights(self):
        cells_to_update = self.internal_active_state.sparse
        self.otp_weights[cells_to_update] += self.otp_lr * (
                self.norm_observation_trace - self.otp_weights[cells_to_update]
        )

    def _get_cells_for_observation(self, obs_state):
        # map observation variables to hidden variables
        cells_in_column = (
                obs_state * self.cells_per_column +
                np.arange(self.cells_per_column, dtype=UINT_DTYPE)
        )
        return cells_in_column

    def _sample_cells(self, messages, shift=0):
        """
            messages.shape = (n_vars, n_states)
        """
        if messages.sum() == 0:
            return np.empty(0, dtype=UINT_DTYPE)

        n_vars, n_states = messages.shape

        next_states = sample_categorical_variables(
            messages,
            self._rng
        )
        # transform states to cell ids
        next_cells = next_states + np.arange(
            0,
            n_states*n_vars,
            n_states
        )
        next_cells += shift

        return next_cells.astype(UINT_DTYPE)
