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
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from hima.common.sdr import sparse_to_dense

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
            num_epochs: int = 10,
            early_stop_loss: float = 0.1,
            retain_old_trajectories: float = 0.5,
            seed=None,
    ):
        torch.set_num_threads(1)

        # n_groups/vars
        self.n_obs_vars = n_obs_vars
        # num of states each obs var has
        self.n_obs_states = n_obs_states
        # total number of obs states
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
        self.num_epochs = num_epochs
        self.early_stop_loss = early_stop_loss
        self.retain_old_trajectories = retain_old_trajectories

        # o_t
        self.observations = list()
        # a_{t-1}
        self.actions = list()
        self.trajectories = list()

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
            self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            # predicted values: logits for further vars-wise softmax application
            self.loss_function = nn.CrossEntropyLoss(reduction='sum')

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        self.loss_propagation_schedule = loss_propagation_schedule
        self._reinit_model_state(reset_loss=True)
        self._reinit_messages_and_states()

    def _reinit_model_state(self, reset_loss: bool):
        self.internal_state = self.get_init_state()
        self.last_loss_value = 0.
        if reset_loss:
            self.accumulated_loss = 0
            self.accumulated_loss_steps = 0

    def _reinit_messages_and_states(self):
        self.internal_forward_messages = self.internal_state
        self.context_messages = self.internal_forward_messages
        self.external_messages = np.zeros(self.external_input_size)

        self.predicted_obs_logits = None
        self.prediction_cells = None
        self.prediction_columns = None

    def get_init_state(self):
        return [
            True,  # Is observed flag
            self.model.get_init_state(),  # Model state
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

        self.trajectories.append(
            list(zip(self.observations, self.actions))
        )

        self.observations.clear()
        self.actions.clear()

        if len(self.trajectories) == self.batch_size:
            self._train()
            self.trajectories = self.trajectories[
                :int(self.batch_size * self.retain_old_trajectories)
            ]

    def observe(self, observation, learn: bool = True):
        if observation.size == self.input_size:
            dense_obs = observation
        else:
            dense_obs = sparse_to_dense(observation, size=self.input_size)
        dense_obs = torch.from_numpy(dense_obs).float().to(self.device)

        if learn:
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
        self.internal_forward_messages = self.internal_state

    def predict(self, learn: bool = False):
        is_observed, state = self.internal_state

        action_probs = None
        if self.external_input_size != 0:
            action_probs = self.external_messages
            action_probs = torch.from_numpy(action_probs).float().to(self.device)

        with torch.set_grad_enabled(learn and not self.use_batches):
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
            from hima.modules.belief.utils import normalize
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

    def _train(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=self.num_epochs
        )
        indx = np.arange(len(self.trajectories))
        self.rng.shuffle(indx)

        split_indx = len(indx) - int(len(indx) * 0.25)

        val_indx = indx[split_indx:]
        train_indx = indx[:split_indx]

        previous_val_loss = 1e24
        tolerance = 0

        for epoch in (pbar := tqdm(range(self.num_epochs))):
            accumulated_loss = 0
            accumulated_steps = 0

            for i in train_indx:
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
            mean_loss = mean_loss.item()

            if len(val_indx) != 0:
                val_los = 0
                val_los_steps = 0

                for i in val_indx:
                    state = self.model.get_init_state()

                    for dense_obs, dense_act in self.trajectories[i]:
                        action_probs = dense_act
                        action_probs = torch.from_numpy(action_probs).float().to(self.device)

                        with torch.no_grad():
                            state = self.transition_with_action(action_probs, state)
                            predicted_obs_logits = self.decode_obs(state)
                            loss = self.get_loss(predicted_obs_logits, dense_obs)
                            val_los += loss
                            val_los_steps += 1
                            state = self.transition_with_observation(dense_obs, state)

                mean_val_loss = (val_los / val_los_steps).item()
            else:
                mean_val_loss = None

            pbar.set_description(
                f"loss: {round(mean_loss, 3)} val_loss: {round(mean_val_loss, 3)} lr: {round(scheduler.get_last_lr()[0], 3)} ",
                refresh=True
            )
            scheduler.step()

            if mean_val_loss < self.early_stop_loss:
                break

            if mean_val_loss > previous_val_loss:
                tolerance += 1
                if tolerance >= 3:
                    break
            else:
                tolerance = 0
                previous_val_loss = mean_val_loss


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

        self.action_repeat_k = self.input_size // self.n_actions // 3
        self.tiled_action_size = self.action_repeat_k * self.action_size

        self.empty_action = torch.zeros((self.tiled_action_size,)).to(self.device)
        self.empty_obs = torch.zeros((self.input_size,)).to(self.device)
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
            layers = [self.full_input_size, 3 * self.n_obs_states]
            self.encoder = nn.Linear(layers[0], layers[1], bias=False)
            encoded_input_size = layers[-1]

        self.lstm = nn.LSTMCell(
            input_size=encoded_input_size,
            hidden_size=self.hidden_size,
            bias=False
        )
        self._initial_state = (
            self.sharpen_out_state(torch.randn(self.hidden_size, device=self.device)),
            torch.randn(self.hidden_size, device=self.device)
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

        state_out, state_cell = self.lstm(obs, state)
        state_out = self.sharpen_out_state(state_out)
        return state_out, state_cell

    def transition_with_action(self, action_probs, state):
        action_probs = action_probs.expand(self.action_repeat_k, -1).flatten()
        obs = torch.cat((self.empty_obs.detach(), action_probs))

        if self.encoder is not None:
            obs = self.encoder(obs)

        state_out, state_cell = self.lstm(obs, state)
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
        return symexp(state_out)

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
            loss = self.loss_function(self.prediction, dense_obs)
            if self.loss is None:
                self.loss = loss
            else:
                self.loss += loss

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

                n_step_dist[step] += 1 / (i + 1) * (prediction - n_step_dist[step])
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


def to_categorical_distributions(logits, n_vars, n_states):
    if n_states == 1:
        # treat it like all vars have binary states --> should sigmoid each var to have prob
        # NB: however now sum(dim=states) != 1, as not(state) is implicit
        return torch.sigmoid(logits)
    else:
        # each var has its own categorical distribution of states obtained with softmax:
        return torch.softmax(
            torch.reshape(logits, (n_vars, n_states)),
            dim=-1
        ).flatten()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    # torch
    return x.detach().cpu().numpy()

