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
from tqdm import tqdm

from dhtm.common.sdr import sparse_to_dense
from dhtm.modules.belief.cortial_column.layer import Layer
from dhtm.modules.baselines.lstm import (
    to_numpy, TLstmLayerHiddenState, TLstmHiddenState,
    to_categorical_distributions, symexp
)
from dhtm.modules.baselines.rwkv_rnn import RwkvCell
from dhtm.modules.belief.utils import normalize


class RwkvLayer(Layer):
    # operational full state, i.e. used internally for any transition
    internal_state: TLstmLayerHiddenState

    # BOTH ARE USED OUTSIDE
    # final full state after any transition
    internal_messages: TLstmLayerHiddenState
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
    accumulated_loss: float | torch.Tensor
    accumulated_loss_steps: int | None
    loss_propagation_schedule: int

    def __init__(
            self,
            n_obs_vars: int,
            n_obs_states: int,
            n_hidden_vars: int,
            n_hidden_states: int,
            n_external_vars: int = 0,
            n_external_states: int = 0,
            lr=2e-3,
            loss_propagation_schedule: int = 5,
            use_batches: bool = True,
            batch_size: int = 50,
            buffer_size: int = 1000,
            num_updates: int = 10,
            early_stop_loss: float = 0.1,
            retain_old_trajectories: float = 0.5,
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
        self.use_batches = use_batches
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_updates = num_updates
        self.early_stop_loss = early_stop_loss
        self.retain_old_trajectories = retain_old_trajectories

        # o_t
        self.observations = list()
        self.observation_messages = np.zeros(self.input_sdr_size)
        # a_{t-1}
        self.actions = list()
        self.trajectories = list()
        self.episodes = 0

        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.with_decoder = not (
            self.n_hidden_vars == self.n_obs_vars
            and self.n_hidden_states == self.n_obs_states
        )
        print(f'RWKV {self.with_decoder=}')

        self.model = RwkvWorldModel(
            n_obs_vars=self.n_obs_vars,
            n_obs_states=self.n_obs_states,
            n_hidden_vars=self.n_hidden_vars,
            n_hidden_states=self.n_hidden_states,
            n_external_vars=self.n_external_vars,
            n_external_states=self.n_external_states,
            with_decoder=self.with_decoder
        ).to(self.device)
        self.model.eval()

        if self.n_obs_states == 1:
            # predicted values: logits for further sigmoid application
            self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            # predicted values: logits for further vars-wise softmax application
            self.loss_function = nn.CrossEntropyLoss(reduction='sum')

        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        self.loss_propagation_schedule = loss_propagation_schedule
        self.mean_loss = 0
        self._reinit_model_state(reset_loss=True)
        self._reinit_messages_and_states()

    def reset_model(self, checkpoint_path=None):
        self.model = RwkvWorldModel(
            n_obs_vars=self.n_obs_vars,
            n_obs_states=self.n_obs_states,
            n_hidden_vars=self.n_hidden_vars,
            n_hidden_states=self.n_hidden_states,
            n_external_vars=self.n_external_vars,
            n_external_states=self.n_external_states,
            with_decoder=self.with_decoder
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()

    def save_model(self, path):
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path
        )

    def _reinit_model_state(self, reset_loss: bool):
        self.internal_state = self.get_init_state()
        self.last_loss_value = 0.
        if reset_loss:
            self.accumulated_loss = 0
            self.accumulated_loss_steps = 0

    def _reinit_messages_and_states(self):
        self.internal_messages = self.internal_state
        self.context_messages = self.internal_messages
        self.external_messages = np.zeros(self.external_input_size)

        self.predicted_obs_logits = None
        self.prediction_cells = None
        self.prediction_columns = None

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
        # should preserve loss from the previous episode
        self._reinit_model_state(reset_loss=False)
        self._reinit_messages_and_states()

        self.episodes += 1
        self.trajectories.append(
            list(zip(self.observations, self.actions))
        )

        self.observations.clear()
        self.actions.clear()

        current_buffer_size = len(self.trajectories)
        if current_buffer_size >= self.batch_size and self.lr > 0:
            if (self.episodes % self.loss_propagation_schedule) == 0:
                self._train()
                self.trajectories = self.trajectories[
                                    -int(current_buffer_size * self.retain_old_trajectories):
                                    ]
                if current_buffer_size > self.buffer_size:
                    self.trajectories = self.trajectories[
                                        -int(self.buffer_size):-1
                                        ]

    def observe(
            self,
            observation: np.array,
            reward: float = 0,
            learn: bool = True
    ):
        if observation.size == self.input_size:
            dense_obs = observation
        else:
            dense_obs = sparse_to_dense(observation, size=self.input_size)
        self.observation_messages = dense_obs
        dense_obs = torch.from_numpy(dense_obs).float().to(self.device)

        if learn and self.lr > 0:
            with torch.set_grad_enabled(not self.use_batches):
                loss = self.get_loss(self.predicted_obs_logits, dense_obs)

            self.last_loss_value = loss.item()

            if self.use_batches:
                self.observations.append(dense_obs)
                self.actions.append(self.external_messages)
            else:
                self.accumulated_loss += loss
                self.accumulated_loss_steps += 1
                self.backpropagate_loss()

        _, state = self.internal_state
        with torch.set_grad_enabled(learn and not self.use_batches):
            state = self.transition_with_observation(dense_obs, state)

        self.internal_state = [True, state]
        self.internal_messages = self.internal_state

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

        self.internal_messages = self.internal_state
        self.prediction_cells = self.internal_messages
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
        if self.accumulated_loss_steps % self.loss_propagation_schedule != 0:
            return

        if self.accumulated_loss_steps > 0:
            self.optimizer.zero_grad()
            mean_loss = self.accumulated_loss / self.accumulated_loss_steps
            mean_loss.backward()
            self.optimizer.step()

        self.accumulated_loss = 0
        self.accumulated_loss_steps = 0
        model_state = self.internal_state[1]
        self.internal_state[1] = (model_state[0].detach(), model_state[1].detach())

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
            self.internal_messages,
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
            self.internal_messages,
            self.external_messages,
            self.context_messages,
            self.predicted_obs_logits,
            self.prediction_cells,
            self.prediction_columns
        ) = snapshot

    def _train(self):
        self.model.train()
        indx = np.arange(len(self.trajectories))

        for update in (pbar := tqdm(range(self.num_updates))):
            accumulated_loss = 0
            accumulated_steps = 0
            self.rng.shuffle(indx)
            for i in indx[:self.batch_size]:
                state = self.model.get_init_state()

                for dense_obs, dense_act in self.trajectories[i]:
                    action_probs = dense_act
                    action_probs = torch.from_numpy(action_probs).float().to(self.device)

                    state = self.transition_with_action(action_probs, state)
                    predicted_obs_logits = self.decode_obs(state)

                    loss = self.get_loss(predicted_obs_logits, dense_obs)
                    accumulated_loss += loss
                    accumulated_steps += 1

                    state = self.transition_with_observation(dense_obs, state)

            self.optimizer.zero_grad()
            mean_loss = accumulated_loss / accumulated_steps
            mean_loss.backward()
            self.optimizer.step()
            self.mean_loss = mean_loss.item()

            pbar.set_description(
                f"loss: {round(self.mean_loss, 3)}",
                refresh=True
            )

            if self.mean_loss < self.early_stop_loss:
                break
        self.model.eval()


class RwkvWorldModel(nn.Module):
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
            self.encoder = None
            encoded_input_size = self.full_input_size

        self.input_projection = nn.Sequential(
            nn.Linear(encoded_input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        self.rwkv = RwkvCell(hidden_size=self.hidden_size)

        rwkv_cell_initial_state = self.rwkv.get_initial_state().to(self.device)
        rwkv_cell_initial_state = (
            rwkv_cell_initial_state +
            torch.randn_like(rwkv_cell_initial_state, device=self.device)
        )
        self._initial_state = (
            self.sharpen_out_state(torch.randn(self.hidden_size, device=self.device)),
            rwkv_cell_initial_state
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
                    nn.SiLU(),
                    nn.Linear(4000, self.input_size, bias=False),
                )
            else:
                self.decoder = nn.Linear(self.hidden_size, self.input_size, bias=False)

    def get_init_state(self) -> TLstmHiddenState:
        return self._initial_state

    def transition_with_observation(self, obs, state):
        if self.action_size > 0:
            obs = torch.cat((obs, self.empty_action.detach()))
        if self.encoder is not None:
            obs = self.encoder(obs)

        x = self.input_projection(obs)
        state_out, state_cell = self.rwkv(x, state)
        state_out = self.sharpen_out_state(state_out)
        return state_out, state_cell

    def transition_with_action(self, action_probs, state):
        action_probs = action_probs.expand(self.action_repeat_k, -1).flatten()
        obs = torch.cat((self.empty_obs.detach(), action_probs))

        if self.encoder is not None:
            obs = self.encoder(obs)

        x = self.input_projection(obs)
        state_out, state_cell = self.rwkv(x, state)
        state_out = self.sharpen_out_state(state_out)
        return state_out, state_cell

    def decode_obs(self, state_out):
        if self.decoder is None:
            return state_out

        state_probs_out = self.to_probabilistic_out_state(state_out)
        obs_logits = self.decoder(state_probs_out)
        obs_logits = self.sharpen_obs_logits(obs_logits)
        return obs_logits

    @staticmethod
    def sharpen_out_state(state_out):
        """Exponentially increase absolute magnitude to reach extreme probabilities."""
        return state_out

    def to_probabilistic_out_state(self, state_out):
        return to_categorical_distributions(
            logits=state_out, n_vars=self.n_hidden_vars, n_states=self.n_hidden_states
        )

    @staticmethod
    def sharpen_obs_logits(obs_logits):
        """Exponentially increase absolute magnitude to reach extreme probabilities."""
        return symexp(obs_logits)

    def to_probabilistic_obs(self, obs_logits):
        return to_categorical_distributions(
            logits=obs_logits, n_vars=self.n_obs_vars, n_states=self.n_obs_states
        )