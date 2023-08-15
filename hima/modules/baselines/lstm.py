#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from hima.common.sdr import sparse_to_dense
from hima.modules.belief.utils import normalize

torch.autograd.set_detect_anomaly(True)


THiddenState = tuple[torch.Tensor, torch.Tensor]


class LstmLayer:
    # layer state
    last_state_snapshot: tuple | None

    # hidden state for making prediction: hidden size
    context_messages: THiddenState
    internal_forward_messages: list[torch.Tensor, torch.Tensor]

    # actions
    external_messages: np.ndarray

    prediction_obs: torch.Tensor | None
    # action-dependent full state: (message[0] * action_probs, message[1])
    prediction_cells: np.ndarray | None
    # predicted observation
    prediction_columns: np.ndarray | None

    loss: torch.Tensor | None

    def __init__(
            self,
            n_obs_vars: int,
            n_obs_states: int,
            n_hidden_vars: int,
            n_hidden_states: int,
            n_external_vars: int = 0,
            n_external_states: int = 0,
            lr=2e-3,
            seed=None,
    ):
        torch.set_num_threads(1)

        # n_groups/vars: 6-10
        self.n_obs_vars = n_obs_vars
        # sps[0]
        self.n_obs_states = n_obs_states
        self.n_columns = self.n_obs_vars * self.n_obs_states

        # actions_dim: 1
        self.n_external_vars = n_external_vars
        # n_actions
        self.n_external_states = n_external_states

        self.n_hidden_vars = n_hidden_vars
        self.n_hidden_states = n_hidden_states

        # context === observation
        self.n_context_vars = self.n_hidden_vars
        self.n_context_states = self.n_hidden_states

        self.input_size = self.n_obs_vars * self.n_obs_states
        self.hidden_size = self.n_hidden_vars * self.n_hidden_states
        self.internal_cells = self.hidden_size
        self.context_input_size = self.hidden_size
        self.external_input_size = self.n_external_vars * self.n_external_states

        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm = LSTMWMUnit(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            external_input_size= self.external_input_size
        ).to(self.device)

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.RMSprop(self.lstm.parameters(), lr=self.lr)

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self._reinit_messages_and_states()

    def _reinit_messages_and_states(self):
        # layer state
        self.last_state_snapshot = None
        self.lstm.message = self.lstm.get_init_message()
        self.context_messages = self.lstm.message
        self.internal_forward_messages = list(self.lstm.message)
        self.external_messages = np.zeros(self.external_input_size)

        self.prediction_obs = None
        self.prediction_cells = None
        self.prediction_columns = None

        self.loss = None

    def reset(self):
        self.optimizer.zero_grad()
        if self.loss is not None:
            self.loss.backward()
            self.optimizer.step()

        self._reinit_messages_and_states()

    def observe(self, observation, learn: bool = True):
        dense_obs = sparse_to_dense(observation, self.input_size, dtype=float)
        dense_obs = torch.from_numpy(dense_obs).float().to(self.device)

        if learn:
            if self.loss is None:
                self.loss = 0
            self.loss += self.loss_function(self.prediction_obs, dense_obs)

        with torch.set_grad_enabled(learn):
            self.lstm.transition_to_next_state(dense_obs)

        self.internal_forward_messages = list(self.lstm.message)

    def predict(self, learn: bool = False):
        action_probs = None
        if self.external_input_size != 0:
            action_probs = torch.from_numpy(self.external_messages).float().to(self.device)
            action_probs = torch.unsqueeze(action_probs, 1)

        with torch.set_grad_enabled(learn):
            self.lstm.apply_action_to_context(action_probs)
            self.prediction_obs = self.lstm.decode_obs()

        self.internal_forward_messages = list(self.lstm.message)
        self.prediction_cells = self.internal_forward_messages

        self.prediction_columns = to_numpy(self.prediction_obs)

    def set_external_messages(self, messages=None):
        # update external cells
        if messages is not None:
            self.external_messages = messages
        elif self.external_input_size != 0:
            self.external_messages = normalize(
                np.zeros(self.external_input_size).reshape(self.n_external_vars, -1)
            ).flatten()

    def set_context_messages(self, messages=None):
        # update context cells
        if messages is not None:
            self.context_messages = messages
            self.lstm.message = tuple(self.context_messages)
        elif self.context_input_size != 0:
            self.context_messages = normalize(
                np.zeros(self.context_input_size).reshape(self.n_context_vars, -1)
            ).flatten()

    def make_state_snapshot(self):
        self.last_state_snapshot = (
            self.lstm.message,
            self.internal_forward_messages,
            self.external_messages,
            self.context_messages,
            self.prediction_obs,
            self.prediction_cells,
            self.prediction_columns
        )

    def restore_last_snapshot(self):
        if self.last_state_snapshot is None:
            return

        (
            self.lstm.message,
            self.internal_forward_messages,
            self.external_messages,
            self.context_messages,
            self.prediction_obs,
            self.prediction_cells,
            self.prediction_columns
        ) = self.last_state_snapshot


class LSTMWMIterative:
    def __init__(
            self,
            n_obs_states,
            n_hidden_states,
            lr=2e-3,
            seed=None
    ):
        self.n_obs_states = n_obs_states
        self.n_hidden_states = n_hidden_states
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm = LSTMWMUnit(
            input_size=n_obs_states,
            hidden_size=n_hidden_states
        ).to(self.device)

        self.prediction = torch.zeros(self.n_obs_states, device=self.device)
        self.loss = None

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.RMSprop(self.lstm.parameters(), lr=self.lr)

        if seed is not None:
            torch.manual_seed(seed)

        self._rng = np.random.default_rng(seed)

    def observe(self, obs, learn=True):
        dense_obs = np.zeros(self.n_obs_states, dtype='float32')
        dense_obs[obs] = 1

        dense_obs = torch.from_numpy(dense_obs).to(self.device)

        if learn:
            if self.loss is None:
                self.loss = self.loss_function(self.prediction, dense_obs)
            else:
                self.loss += self.loss_function(self.prediction, dense_obs)

            self.prediction = self.lstm(dense_obs)
        else:
            with torch.no_grad():
                self.prediction = self.lstm(dense_obs)

    def reset(self):
        self.optimizer.zero_grad()

        if self.loss is not None:
            self.loss.backward()
            self.optimizer.step()
            self.loss = None

        self.prediction = torch.zeros(self.n_obs_states, device=self.device)
        self.lstm.message = (
            torch.zeros(self.n_hidden_states, device=self.device),
            torch.zeros(self.n_hidden_states, device=self.device),
        )

    def n_step_prediction(self, initial_dist, steps, mc_iterations=100):
        n_step_dist = np.zeros((steps, self.n_obs_states))
        initial_message = (self.lstm.message[0].clone(), self.lstm.message[1].clone())

        for i in range(mc_iterations):
            dist_curr_step = initial_dist
            for step in range(steps):
                # sample observation from prediction density
                gamma = self._rng.random(size=self.n_obs_states)
                obs = np.flatnonzero(gamma < dist_curr_step)
                dense_obs = np.zeros(self.n_obs_states, dtype='float32')
                dense_obs[obs] = 1
                dense_obs = torch.from_numpy(dense_obs).to(self.device)

                # predict distribution
                with torch.no_grad():
                    prediction = self.lstm(dense_obs).cpu().detach().numpy()

                n_step_dist[step] += 1/(i+1) * (prediction - n_step_dist[step])
                dist_curr_step = prediction

            self.lstm.message = initial_message

        return n_step_dist


class LSTMWMUnit(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            external_input_size=0
    ):
        super(LSTMWMUnit, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_obs_states = input_size
        self.n_actions = external_input_size
        self.n_hidden_states = hidden_size
        self.full_hidden_size = self.n_hidden_states * max(self.n_actions, 1)

        self.lstm = nn.LSTMCell(
            input_size=self.n_obs_states,
            hidden_size=self.full_hidden_size
        )

        # The linear layer that maps from hidden state space back to obs space
        self.hidden2obs = nn.Linear(
            self.n_hidden_states,
            self.n_obs_states
        )

        self.message = self.get_init_message()

    def get_init_message(self) -> THiddenState:
        return (
            torch.zeros(self.full_hidden_size, device=self.device),
            torch.zeros(self.full_hidden_size, device=self.device)
        )

    def transition_to_next_state(self, obs):
        self.message = self.lstm(obs, self.message)
        return self.message

    def apply_action_to_context(self, action_probs):
        if self.n_actions <= 1:
            return

        msg = self.message[0]

        msg = msg.reshape(self.n_actions, -1)
        msg = msg * action_probs
        msg = msg.flatten()

        self.message = msg, self.message[1]

    def decode_obs(self):
        obs_msg = self.message[0]

        if self.n_actions > 1:
            obs_msg = torch.sum(obs_msg.reshape(self.n_actions, -1), dim=0)

        prediction_logit = self.hidden2obs(obs_msg)
        prediction = torch.sigmoid(prediction_logit)
        return prediction

    def forward(self, obs):
        self.transition_to_next_state(obs)
        return self.decode_obs()


class LSTMWMLayer(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            n_layers=1,
            dropout=0.2
    ):
        super(LSTMWMLayer, self).__init__()

        self.n_obs_states = input_size
        self.n_hidden_states = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

        # The linear layer that maps from hidden state space back to obs space
        self.hidden2obs = nn.Linear(
            hidden_size,
            input_size
        )

        self.message = (
            torch.zeros(self.n_hidden_states),
            torch.zeros(self.n_hidden_states)
        )

    def forward(self, obs):
        hidden, self.message = self.lstm(obs, self.message)
        prediction_logit = self.hidden2obs(hidden)
        prediction = torch.sigmoid(prediction_logit)
        return prediction


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    # torch
    return x.detach().cpu().numpy()
