#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np

from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from hima.modules.belief.utils import normalize, softmax, sample_categorical_variables
from hima.experiments.hmm.runners.utils import get_surprise


class BioHIMA:
    def __init__(
            self,
            cortical_column: CorticalColumn,
            gamma: float = 0.99,
            observation_prior_lr: float = 1.0,
            striatum_lr: float = 1.0,
            sr_steps: int = 5,
            approximate_tail: bool = True,
            inverse_temp: float = 1.0,
            seed: int = None
    ):
        self.observation_prior_lr = observation_prior_lr
        self.striatum_lr = striatum_lr
        self.cortical_column = cortical_column
        self.gamma = gamma
        self.sr_steps = sr_steps
        self.approximate_tail = approximate_tail
        self.inverse_temp = inverse_temp

        self.observation_prior = normalize(
            np.ones(
                (
                    self.cortical_column.layer.n_obs_vars,
                    self.cortical_column.layer.n_obs_states
                )
            )
        ).flatten()

        self.observation_messages = self.observation_prior.copy()

        self.striatum_weights = np.zeros(
            (
                (
                    self.cortical_column.layer.n_hidden_states *
                    self.cortical_column.layer.n_hidden_vars
                 ),
                (
                    self.cortical_column.layer.n_obs_states *
                    self.cortical_column.layer.n_obs_vars
                )
            )
        )

        self.surprise = 0
        self.td_error = 0

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def sample_action(self):
        """
        Evaluate and sample actions
        """
        self.cortical_column.make_state_snapshot()

        action_values = np.zeros(self.cortical_column.layer.external_input_size)

        for action in range(self.cortical_column.layer.external_input_size):
            dense_action = np.zeros_like(action_values)
            dense_action[action] = 1

            self.cortical_column.predict(
                self.cortical_column.layer.context_messages,
                dense_action
            )

            sr = self._generate_sr(
                self.sr_steps,
                self.cortical_column.layer.prediction_cells,
                self.cortical_column.layer.prediction_columns,
                save_state=False
            )

            self.cortical_column.restore_last_snapshot()

            action_values[action] = np.sum(
                sr * self.observation_prior
            )

        action_dist = softmax(action_values, beta=self.inverse_temp)
        action = sample_categorical_variables(action_dist.reshape((1, -1)), self._rng)

        return action[0]

    def observe(self, observation, learn=True):
        """
        Main learning routine
            observation: tuple (image, action)
                events: sparse sdr
                action: sparse sdr
        """
        events, action = observation
        # predict current events using observed action
        self.cortical_column.observe(events, action, learn=learn)

        if len(self.cortical_column.output_sdr.sparse) > 0:
            self.surprise = get_surprise(
                self.cortical_column.layer.prediction_columns,
                self.cortical_column.output_sdr.sparse,
                mode='categorical'
            )
        else:
            self.surprise = 0

        self.observation_messages = np.zeros_like(
            self.observation_messages
        )
        self.observation_messages[self.cortical_column.output_sdr.sparse] = 1

        # striatum TD learning
        if learn:
            prediction_cells = self.cortical_column.layer.prediction_cells.copy()

            predicted_sr = self.predict_sr(prediction_cells)
            generated_sr, last_step_prediction = self._generate_sr(
                self.sr_steps,
                prediction_cells,
                self.observation_messages,
                approximate_tail=self.approximate_tail,
                return_last_prediction_step=True
            )

            delta_sr = generated_sr - predicted_sr
            delta_w = prediction_cells.reshape((-1, 1)) * delta_sr.reshape((1, -1))

            self.striatum_weights += self.striatum_lr * delta_w

            self.striatum_weights = np.clip(self.striatum_weights, 0, None)

            self.td_error = np.sum(np.power(delta_sr, 2))

            return predicted_sr, generated_sr

    def reinforce(self, reward):
        """
        Adapt prior distribution of observations according to external reward.
            reward: float in [0, 1]
        """
        self.observation_prior += reward * self.observation_prior_lr * self.observation_messages
        self.observation_prior = normalize(
            self.observation_prior.reshape((self.cortical_column.layer.n_obs_vars, -1))
        ).flatten()

    def reset(self, initial_context_message, initial_external_message):
        self.cortical_column.reset(initial_context_message, initial_external_message)

    def _generate_sr(
            self,
            n_steps,
            initial_messages,
            initial_prediction,
            approximate_tail=True,
            return_last_prediction_step=False,
            save_state=True
    ):
        """
        Policy is assumed to be uniform.
        """
        if save_state:
            self.cortical_column.make_state_snapshot()

        sr = np.zeros_like(self.observation_prior)

        predicted_observation = initial_prediction
        context_messages = initial_messages

        i = -1
        for i in range(n_steps):
            sr += (self.gamma**i) * predicted_observation

            self.cortical_column.predict(
                context_messages
            )

            predicted_observation = self.cortical_column.layer.prediction_columns.copy()
            context_messages = self.cortical_column.layer.internal_forward_messages.copy()

        if approximate_tail:
            sr += (self.gamma**(i+1)) * self.predict_sr(
                context_messages
            )

        if save_state:
            self.cortical_column.restore_last_snapshot()

        if return_last_prediction_step:
            return sr, context_messages
        else:
            return sr

    def predict_sr(self, hidden_vars_dist):
        sr = np.dot(hidden_vars_dist, self.striatum_weights)
        sr /= self.cortical_column.layer.n_hidden_vars
        return sr
