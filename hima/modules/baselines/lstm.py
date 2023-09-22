#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hima.common.sdr import sparse_to_dense
from hima.modules.belief.utils import normalize

TLstmHiddenState = tuple[torch.Tensor, torch.Tensor]

# the bool variable describes lstm current hidden state: True — observed, False — predicted
TLstmLayerHiddenState = list[bool, TLstmHiddenState]


class LstmLayer:
    # operational full state, i.e. used internally for any transition
    internal_state: TLstmLayerHiddenState

    # BOTH ARE USED OUTSIDE
    # final full state after any transition
    internal_forward_messages: TLstmLayerHiddenState
    # passed full state for prediction
    context_messages: TLstmLayerHiddenState

    # actions
    external_messages: np.ndarray | None

    # predicted decoded observation
    predicted_obs_logits: torch.Tensor | None
    # numpy copy of prediction_obs
    prediction_columns: np.ndarray | None

    # copy of internal_forward_messages
    prediction_cells: np.ndarray | None

    # value particularly for the last step
    last_loss_value: float
    accumulated_loss: torch.Tensor | None
    accumulated_loss_steps: int | None

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
        # num of states each obs var has
        self.n_obs_states = n_obs_states
        # full number of obs states
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
        self.input_sdr_size = self.input_size

        self.hidden_size = self.n_hidden_vars * self.n_hidden_states
        self.internal_cells = self.hidden_size
        self.context_input_size = self.hidden_size
        self.external_input_size = self.n_external_vars * self.n_external_states

        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        with_decoder = not (
            self.n_hidden_vars == self.n_obs_vars
            and self.n_hidden_states == self.n_obs_states
        )
        print(f'LSTM {with_decoder=}')

        self.model = LstmWorldModel(
            n_obs_vars=n_obs_vars,
            n_obs_states=n_obs_states,
            n_hidden_vars=n_hidden_vars,
            n_hidden_states=n_hidden_states,
            n_external_vars=n_external_vars,
            n_external_states=n_external_states,
            with_decoder=with_decoder
        ).to(self.device)

        if self.n_obs_states == 1:
            # predicted values: logits for further sigmoid application
            self.loss_function = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            # predicted values: logits for further vars-wise softmax application
            self.loss_function = nn.CrossEntropyLoss(reduction='sum', label_smoothing=1e-3)

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        self._reinit_messages_and_states()

    def _reinit_messages_and_states(self):
        # layer state
        self.internal_state = self.get_init_state()
        self.internal_forward_messages = self.internal_state
        self.context_messages = self.internal_forward_messages
        self.external_messages = np.zeros(self.external_input_size)

        self.predicted_obs_logits = None
        self.prediction_cells = None
        self.prediction_columns = None

        self.accumulated_loss = None
        self.accumulated_loss_steps = None
        self.last_loss_value = 0.

    def get_init_state(self):
        return [
            True,                           # Is observed flag
            self.model.get_init_state(),    # Model state
        ]

    def transition_with_observation(self, obs, state):
        return self.model.transition_with_observation(obs, state)

    def transition_with_action(self, action_probs, state):
        return self.model.transition_with_action(action_probs, state)

    def decode_obs(self, state):
        state_out, _ = state
        return self.model.decode_obs(state_out)

    def reset(self):
        self.optimizer.zero_grad()
        if self.accumulated_loss is not None:
            mean_loss = self.accumulated_loss / self.accumulated_loss_steps
            mean_loss.backward()
            self.optimizer.step()

        self._reinit_messages_and_states()

    def observe(self, observation, learn: bool = True):
        dense_obs = sparse_to_dense(observation, size=self.input_size)
        dense_obs = torch.from_numpy(dense_obs).float().to(self.device)

        if learn:
            if self.accumulated_loss is None:
                self.accumulated_loss = 0
                self.accumulated_loss_steps = 0
            loss = self.get_loss(self.predicted_obs_logits, dense_obs)
            self.last_loss_value = loss.item()
            self.accumulated_loss += loss
            self.accumulated_loss_steps += 1

        _, state = self.internal_state
        with torch.set_grad_enabled(learn):
            state = self.transition_with_observation(dense_obs, state)

        self.internal_state = [True, state]
        self.internal_forward_messages = self.internal_state

    def predict(self, learn: bool = False):
        is_observed, state = self.internal_state

        action_probs = None
        if self.external_input_size != 0:
            action_probs = self.external_messages
            action_probs = torch.from_numpy(action_probs).float().to(self.device)

        if not learn and not is_observed:
            # PLANNING:
            # should observe what was predicted previously before making new prediction
            with torch.no_grad():
                state = self.transition_with_observation(self.predicted_obs_logits, state)

        with torch.set_grad_enabled(learn):
            if self.external_input_size != 0:
                state = self.transition_with_action(action_probs, state)
            self.predicted_obs_logits = self.decode_obs(state)

        self.internal_state = [False, state]

        self.internal_forward_messages = self.internal_state
        self.prediction_cells = self.internal_forward_messages
        self.prediction_columns = to_numpy(
            self.model.as_probabilistic_obs(self.predicted_obs_logits.detach())
        )

    def get_loss(self, logits, target):
        if self.n_obs_states == 1:
            # BCE with logits
            return self.loss_function(logits, target)
        else:
            # calculate cross entropy over each variable
            # for it, we reshape as if it is a batch of distributions
            shape = self.n_obs_vars, self.n_obs_states
            logits = torch.unsqueeze(torch.reshape(logits, shape).T, 0)
            target = torch.unsqueeze(torch.reshape(target, shape).T, 0)
            return self.loss_function(logits, target)

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
            self.internal_state = messages
        elif self.context_input_size != 0:
            assert False, f"Below is incorrect, implement it!"
            # self.context_messages = normalize(
            #     np.zeros(self.context_input_size).reshape(self.n_context_vars, -1)
            # ).flatten()

    def make_state_snapshot(self):
        return (
            # mutable attributes:

            # immutable attributes:
            self.internal_state,
            self.internal_forward_messages,
            self.external_messages,
            self.context_messages,
            self.predicted_obs_logits,
            self.prediction_cells,
            self.prediction_columns
        )

    def restore_last_snapshot(self, snapshot):
        if snapshot is None:
            return

        (
            self.internal_state,
            self.internal_forward_messages,
            self.external_messages,
            self.context_messages,
            self.predicted_obs_logits,
            self.prediction_cells,
            self.prediction_columns
        ) = snapshot


class LstmWorldModel(nn.Module):
    def __init__(
            self,
            n_obs_vars: int,
            n_obs_states: int,
            n_hidden_vars: int,
            n_hidden_states: int,
            n_external_vars: int = 0,
            n_external_states: int = 0,
            with_decoder: bool = True
    ):
        super(LstmWorldModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_obs_vars = n_obs_vars
        self.n_obs_states = n_obs_states
        self.input_size = self.n_obs_vars * self.n_obs_states

        self.n_actions = n_external_vars
        self.n_action_states = n_external_states
        self.action_size = self.n_actions * self.n_action_states

        self.n_hidden_vars = n_hidden_vars
        self.n_hidden_states = n_hidden_states
        self.hidden_size = self.n_hidden_vars * self.n_hidden_states

        pinball_raw_image = self.n_obs_vars == 50 * 36 and self.n_obs_states == 1
        if pinball_raw_image:
            self.encoder = nn.Sequential(
                nn.Unflatten(0, (1, 50, 36)),
                # 50x36x1
                nn.Conv2d(1, 4, 5, 3, 2),
                # 17x11x2
                nn.Conv2d(4, 8, 5, 2, 2),
                # 9x6x4
                # nn.Conv2d(8, 8, 3, 1, 1),
                # 9x6x4
                nn.Flatten(0),
                # 432
            )
            encoded_input_size = 432
        else:
            self.encoder = None
            encoded_input_size = self.input_size

        self.state_lstm = nn.LSTMCell(
            input_size=encoded_input_size,
            hidden_size=self.hidden_size
        )

        if self.action_size > 0:
            # self.action_projection = nn.Linear()
            self.action_lstm = nn.LSTMCell(
                input_size=self.action_size,
                hidden_size=self.hidden_size
            )

        self.decoder = None
        if with_decoder:
            # maps from hidden state space back to obs space
            if pinball_raw_image:
                # Pinball raw image decoder
                self.decoder = nn.Sequential(
                    nn.Linear(self.hidden_size, 1000),
                    nn.SiLU(),
                    nn.Linear(1000, 4000),
                    nn.Tanh(),
                    nn.Linear(4000, self.input_size, bias=False),
                )
            else:
                # NB: single linear layer is tested to work better than MLP
                self.decoder = nn.Sequential(
                    nn.Linear(self.hidden_size, self.input_size, bias=False),
                )

    def get_init_state(self) -> TLstmHiddenState:
        return (
            torch.zeros(self.hidden_size, device=self.device),
            torch.zeros(self.hidden_size, device=self.device)
        )

    def transition_with_observation(self, obs, state):
        if self.encoder is not None:
            obs = self.encoder(obs)
        return self.state_lstm(obs, state)

    def transition_with_action(self, action_probs, state):
        return self.action_lstm(action_probs, state)

    def decode_obs(self, state_out):
        if self.decoder is None:
            return state_out

        state_probs_out = self.as_probabilistic_out(state_out)
        obs_logit = self.decoder(state_probs_out)
        return obs_logit

    def as_probabilistic_out(self, state_out):
        return as_distributions(
            logits=state_out, n_vars=self.n_hidden_vars, n_states=self.n_hidden_states
        )

    def as_probabilistic_obs(self, obs_logits):
        return as_distributions(
            logits=obs_logits, n_vars=self.n_obs_vars, n_states=self.n_obs_states
        )


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
        self.lstm.message = self.lstm.get_init_message()

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
            external_input_size: int = 0,
            decoder_bias: bool = True
    ):
        super(LSTMWMUnit, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_obs_states = input_size
        self.n_actions = external_input_size
        self.n_hidden_states = hidden_size

        self.lstm = nn.LSTMCell(
            input_size=self.n_obs_states,
            hidden_size=self.full_hidden_size
        )

        # The linear layer that maps from hidden state space back to obs space
        self.hidden2obs = nn.Linear(
            self.n_hidden_states,
            self.n_obs_states,
            bias=decoder_bias
        )

        self.message = self.get_init_message()

    def get_init_message(self) -> TLstmHiddenState:
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


def as_distributions(logits, n_vars, n_states):
    if n_states == 1:
        # treat it like all vars have binary states --> should sigmoid each var to have prob
        # NB: however now sum(dim=states) != 1, as not(state) is implicit
        return torch.sigmoid(logits)
    else:
        # each var has its own distribution of states obtained with softmax:
        return torch.softmax(
            torch.reshape(logits, (n_vars, n_states)),
            dim=1
        ).flatten()


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    # torch
    return x.detach().cpu().numpy()
