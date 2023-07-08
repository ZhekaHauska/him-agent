#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np

from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from hima.modules.belief.utils import normalize, softmax, sample_categorical_variables


class BioHIMA:
    def __init__(
            self,
            cortical_column: CorticalColumn,
            gamma: float = 0.99,
            observation_prior_lr: float = 1.0,
            sr_steps: int = 5,
            inverse_temp: float = 1.0,
            seed: int = None
    ):
        self.observation_prior_lr = observation_prior_lr
        self.cortical_column = cortical_column
        self.gamma = gamma
        self.sr_steps = sr_steps
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

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def sample_action(self):
        """
        Evaluate and sample actions
        """
        action_values = np.zeros(self.cortical_column.layer.external_input_size)

        for action in range(self.cortical_column.layer.external_input_size):
            dense_action = np.zeros_like(action_values)
            dense_action[action] = 1

            sr = self._generate_sr(
                self.sr_steps,
                self.cortical_column.layer.context_messages,
                dense_action
            )

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
                image: sparse sdr
                action: sparse sdr
        """
        image, action = observation
        self.cortical_column.observe(image, action, learn=learn)

        self.observation_messages = np.zeros_like(
            self.observation_messages
        )
        self.observation_messages[self.cortical_column.output_sdr.sparse] = 1

    def reinforce(self, reward):
        """
        Adapt prior distribution of observations according to external reward.
            reward: float in [0, 1]
        """
        self.observation_prior += reward * self.observation_prior_lr * self.observation_messages
        self.observation_prior = normalize(
            self.observation_prior.reshape((self.cortical_column.layer.n_obs_vars, -1))
        ).flatten()

    def _generate_sr(self, n_steps, context_messages, external_messages, approximate_tail=True):
        """
        Generate SR controlling only the first action.
        Further policy is assumed to be uniform.
        """
        sr = np.zeros_like(self.observation_prior)

        self.cortical_column.predict(
            context_messages,
            external_messages
        )

        i = -1
        for i in range(n_steps):
            sr += (self.gamma**i) * self.cortical_column.predicted_observation

            context_messages = self.cortical_column.layer.internal_forward_messages

            self.cortical_column.predict(
                context_messages
            )

        if approximate_tail:
            sr += (self.gamma**(i+1)) * self._predict_sr(
                self.cortical_column.layer.internal_forward_messages
            )

        return sr

    def _predict_sr(self, hidden_vars_dist):
        sr = np.dot(hidden_vars_dist, self.striatum_weights)
        sr /= self.cortical_column.layer.n_hidden_vars
        sr = (1 - self.gamma) * sr.reshape((self.cortical_column.layer.n_obs_vars, -1))
        sr = normalize(sr).flatten()
        sr /= (1 - self.gamma)

        return sr
