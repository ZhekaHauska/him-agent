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
from hima.modules.baselines.lstm import (
    to_numpy, TLstmLayerHiddenState, TLstmHiddenState,
    to_categorical_distributions
)
from hima.modules.baselines.rwkv_rnn import RwkvCell
from hima.modules.belief.utils import normalize

torch.autograd.set_detect_anomaly(True)


class RwkvLayer:
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
        print(f'RWKV {with_decoder=}')

        self.model = RwkvWorldModel(
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
            self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            # predicted values: logits for further vars-wise softmax application
            self.loss_function = nn.CrossEntropyLoss(reduction='sum')

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
        self.backpropagate_loss()
        self._reinit_messages_and_states()

    def observe(self, observation, learn: bool = True):
        if observation.size == self.input_size:
            dense_obs = observation
        else:
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
            # if self.accumulated_loss_steps % 5 == 0:
            #     self.backpropagate_loss()

        _, state = self.internal_state
        with torch.set_grad_enabled(learn):
            state = self.transition_with_observation(dense_obs, state)

        self.internal_state = [True, state]
        # self.internal_state = [
        #     self.internal_state[0],
        #     (self.internal_state[1][0].detach(), self.internal_state[1][1].detach())
        # ]
        self.internal_forward_messages = self.internal_state

    def predict(self, learn: bool = False):
        is_observed, state = self.internal_state

        action_probs = None
        if self.external_input_size != 0:
            action_probs = self.external_messages
            action_probs = torch.from_numpy(action_probs).float().to(self.device)

        with torch.set_grad_enabled(learn):
            if self.external_input_size != 0:
                state = self.transition_with_action(action_probs, state)
            self.predicted_obs_logits = self.decode_obs(state)

        self.internal_state = [False, state]

        self.internal_forward_messages = self.internal_state
        self.prediction_cells = self.internal_forward_messages
        self.prediction_columns = to_numpy(
            self.model.to_probabilistic_obs(self.predicted_obs_logits.detach())
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
            return self.loss_function(logits, target) / self.n_obs_vars

    def backpropagate_loss(self):
        if self.accumulated_loss is None:
            return

        self.optimizer.zero_grad()
        mean_loss = self.accumulated_loss / self.accumulated_loss_steps
        mean_loss.backward()
        self.optimizer.step()

        self.accumulated_loss = None
        self.internal_state = [
            self.internal_state[0],
            (self.internal_state[1][0].detach(), self.internal_state[1][1].detach())
        ]

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


class RwkvWorldModel(nn.Module):
    out_temp: float = 1.0
    obs_temp: float = 1.0

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
        super(RwkvWorldModel, self).__init__()
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

        self.action_repeat_k = self.input_size // self.n_actions // 3
        self.tiled_action_size = self.action_repeat_k * self.action_size

        self.empty_action = torch.zeros((self.tiled_action_size, )).to(self.device)
        self.empty_obs = torch.zeros((self.input_size, )).to(self.device)
        self.full_input_size = self.input_size + self.tiled_action_size

        pinball_raw_image = self.n_obs_vars == 50 * 36 and self.n_obs_states == 1
        if pinball_raw_image:
            self.encoder = nn.Sequential(
                nn.Unflatten(0, (1, 50, 36)),
                # 50x36x1
                nn.Conv2d(1, 4, 5, 3, 2),
                # 17x11x2
                nn.Conv2d(4, 4, 5, 2, 2),
                # 9x6x4
                # nn.Conv2d(4, 8, 3, 1, 1),
                # 9x6x4
                nn.Flatten(0),
            )
            encoded_input_size = 216
        else:
            # self.encoder = None
            # encoded_input_size = self.full_input_size
            layers = [self.full_input_size, 3 * self.n_obs_states, self.hidden_size]
            self.encoder = nn.Sequential(
                nn.Linear(layers[0], layers[1], bias=False),
                nn.SiLU(),
                nn.Linear(layers[1], layers[2], bias=True),
                nn.Tanh(),
            )
            encoded_input_size = layers[-1]

        self.input_projection = nn.Sequential(
            nn.Linear(encoded_input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        self.rwkv = RwkvCell(hidden_size=self.hidden_size)

        self.decoder = None
        if with_decoder:
            # maps from hidden state space back to obs space
            if pinball_raw_image:
                # Pinball raw image decoder
                self.decoder = nn.Sequential(
                    nn.Linear(self.hidden_size, 1000),
                    nn.SiLU(),
                    nn.Linear(1000, 4000),
                    nn.SiLU(),
                    nn.Linear(4000, self.input_size, bias=False),
                )
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size, self.input_size, bias=False),
                )

    def get_init_state(self) -> TLstmHiddenState:
        return (
            torch.zeros(self.hidden_size, device=self.device),
            self.rwkv.get_initial_state().to(self.device)
        )

    def transition_with_observation(self, obs, state):
        if self.action_size > 0:
            obs = torch.cat((obs, self.empty_action.detach()))
        if self.encoder is not None:
            obs = self.encoder(obs)
        x = self.input_projection(obs)
        state_out, state_cell = self.rwkv(x, state)
        # increase temperature for out state
        return state_out * self.out_temp, state_cell

    def transition_with_action(self, action_probs, state):
        action_probs = action_probs.expand(self.action_repeat_k, -1).flatten()
        obs = torch.cat((self.empty_obs.detach(), action_probs))

        if self.encoder is not None:
            obs = self.encoder(obs)
        x = self.input_projection(obs)
        state_out, state_cell = self.rwkv(x, state)
        # increase temperature for out state
        return state_out * self.out_temp, state_cell

    def decode_obs(self, state_out):
        if self.decoder is None:
            return state_out

        state_probs_out = self.to_probabilistic_out_state(state_out)
        obs_logit = self.decoder(state_probs_out)
        return obs_logit

    def to_probabilistic_out_state(self, state_out):
        return to_categorical_distributions(
            logits=state_out, n_vars=self.n_hidden_vars, n_states=self.n_hidden_states
        )

    def to_probabilistic_obs(self, obs_logits):
        return to_categorical_distributions(
            logits=obs_logits, n_vars=self.n_obs_vars, n_states=self.n_obs_states
        )
