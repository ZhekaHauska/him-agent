#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np

from hima.common.sdr import sparse_to_dense
from hima.experiments.hmm.runners.utils import get_surprise
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from hima.modules.belief.utils import normalize, softmax


class BioHIMA:
    def __init__(
            self,
            cortical_column: CorticalColumn,
            *,
            gamma: float = 0.99,
            observation_reward_lr: float = 0.01,
            striatum_lr: float = 1.0,
            sr_steps: int = 5,
            approximate_tail: bool = True,
            inverse_temp: float = 1.0,
            reward_scale: float = 1.0,
            exploration_eps: float = -1,
            seed: int = None
    ):
        self.observation_reward_lr = observation_reward_lr
        self.striatum_lr = striatum_lr
        self.cortical_column = cortical_column
        self.gamma = gamma
        self.sr_steps = sr_steps
        self.approximate_tail = approximate_tail
        self.inverse_temp = inverse_temp
        self.reward_scale = reward_scale
        self.exploration_eps = exploration_eps

        layer = self.cortical_column.layer
        observation_rewards = np.zeros((layer.n_obs_vars, layer.n_obs_states))
        self.observation_rewards = observation_rewards.flatten()
        self.observation_prior = self.get_observations_prior(self.observation_rewards)
        self.observation_messages = self.observation_prior.copy()

        self.striatum_weights = np.zeros((
            (layer.n_hidden_states * layer.n_hidden_vars),
            (layer.n_obs_states * layer.n_obs_vars)
        ))

        self.surprise = 0
        self.td_error = 0

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, initial_context_message, initial_external_message):
        self.cortical_column.reset(initial_context_message, initial_external_message)

    def sample_action(self):
        """Evaluate and sample actions."""
        action_values = self.evaluate_actions()
        action = self._select_action(action_values)
        return action

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
        encoded_obs = self.cortical_column.output_sdr.sparse

        self.surprise = 0
        if len(encoded_obs) > 0:
            self.surprise = get_surprise(
                self.cortical_column.layer.prediction_columns, encoded_obs, mode='categorical'
            )

        self.observation_messages = sparse_to_dense(encoded_obs, like=self.observation_messages)
        if not learn:
            return
        # striatum TD learning

        prediction_cells = self.cortical_column.layer.prediction_cells

        predicted_sr = self.predict_sr(prediction_cells)
        generated_sr = self._generate_sr(
            self.sr_steps,
            initial_messages=prediction_cells,
            initial_prediction=self.observation_messages,
            approximate_tail=self.approximate_tail,
        )
        self.td_update_sr(generated_sr, predicted_sr, prediction_cells)
        return predicted_sr, generated_sr

    def reinforce(self, reward):
        """
        Adapt prior distribution of observations according to external reward.
            reward: float in [0, 1]
        """
        # learn with mse loss
        lr, messages = self.observation_reward_lr, self.observation_messages

        self.observation_rewards += lr * messages * (reward - self.observation_rewards)
        self.observation_prior = self.get_observations_prior(self.observation_rewards)

    def evaluate_actions(self):
        """Evaluate Q[s,a] for each action."""
        self.cortical_column.make_state_snapshot()

        n_actions = self.cortical_column.layer.external_input_size
        action_values = np.zeros(n_actions)
        dense_action = np.zeros_like(action_values)

        for action in range(n_actions):
            # hacky way to clean previous one-hot, for 0th does nothing
            dense_action[action-1] = 0
            # set current one-hot
            dense_action[action] = 1

            self.cortical_column.predict(
                context_messages=self.cortical_column.layer.context_messages,
                external_messages=dense_action
            )
            # TODO switch to predict_sr when TD is low
            sr = self._generate_sr(
                self.sr_steps,
                initial_messages=self.cortical_column.layer.prediction_cells,
                initial_prediction=self.cortical_column.layer.prediction_columns,
                save_state=False
            )

            self.cortical_column.restore_last_snapshot()

            action_values[action] = np.sum(
                sr * np.log(np.clip(self.observation_prior, 1e-7, 1))
            ) / self.cortical_column.layer.n_obs_vars

        return action_values

    def _generate_sr(
            self,
            n_steps,
            initial_messages,
            initial_prediction,
            approximate_tail=True,
            return_last_prediction_step=False,
            save_state=True
    ):
        # NB: Policy is assumed to be uniform.
        if save_state:
            self.cortical_column.make_state_snapshot()

        sr = np.zeros_like(self.observation_prior)

        context_messages = initial_messages
        predicted_observation = initial_prediction

        t = -1
        for t in range(n_steps):
            sr += predicted_observation * self.gamma**t

            self.cortical_column.predict(context_messages)

            context_messages = self.cortical_column.layer.internal_forward_messages.copy()
            predicted_observation = self.cortical_column.layer.prediction_columns

        if approximate_tail:
            sr += self.predict_sr(context_messages) * self.gamma**(t+1)

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

    def td_update_sr(self, target_sr, predicted_sr, prediction_cells):
        error_sr = target_sr - predicted_sr
        # dSR / dW for linear model
        delta_w = np.outer(prediction_cells, error_sr)

        self.striatum_weights += self.striatum_lr * delta_w
        # FIXME: why does it need to clip negatives?
        self.striatum_weights = np.clip(self.striatum_weights, 0, None)

        self.td_error = np.sum(np.power(error_sr, 2))

    def get_observations_prior(self, rewards):
        scale = self.reward_scale
        rewards = rewards.reshape(self.cortical_column.layer.n_obs_vars, -1)
        # FIXME: should it be `softmax(scale * rewards, axis=-1)`?
        return normalize(np.exp(scale * rewards)).flatten()

    def _select_action(self, action_values):
        n_actions = self.cortical_column.layer.external_input_size

        if self.exploration_eps < 0:
            # softmax policy
            action_dist = softmax(action_values, beta=self.inverse_temp)
            action = self._rng.choice(n_actions, p=action_dist)
        else:
            # eps-greedy policy
            if self._rng.random() < self.exploration_eps:
                # random
                action = self._rng.choice(n_actions)
            else:
                action = np.argmax(action_values)
        return action
