#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from collections import deque
from enum import Enum, auto

import numpy as np

from hima.common.sdr import sparse_to_dense
from hima.common.utils import softmax, lin_sum, safe_divide
from hima.experiments.hmm.runners.utils import get_surprise
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn


class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()


class SrEstimatePlanning(Enum):
    UNIFORM = 1
    ON_POLICY = auto()
    OFF_POLICY = auto()


class ActionValueEstimate(Enum):
    PREDICT = 1
    PLAN = auto()
    BALANCE = auto()


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
            seed: int = None,
            exploration_eps: float = -1,
            action_value_estimate: str = 'plan',
            sr_estimate_planning: str = 'uniform',
    ):
        self.observation_reward_lr = observation_reward_lr
        self.striatum_lr = striatum_lr
        self.cortical_column = cortical_column
        self.gamma = gamma
        self.sr_steps = sr_steps
        self.approximate_tail = approximate_tail
        self.inverse_temp = inverse_temp
        self.reward_scale = reward_scale

        if exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = exploration_eps

        self.action_value_estimate = ActionValueEstimate[action_value_estimate.upper()]
        self.sr_estimate_planning = SrEstimatePlanning[sr_estimate_planning.upper()]

        layer = self.cortical_column.layer
        observation_rewards = np.zeros((layer.n_obs_vars, layer.n_obs_states))
        self.observation_rewards = observation_rewards.flatten()
        self.observation_messages = np.zeros_like(self.observation_rewards)

        self.striatum_weights = np.zeros((
            (layer.n_hidden_states * layer.n_hidden_vars),
            (layer.n_obs_states * layer.n_obs_vars)
        ))

        self.state_snapshot_stack = deque()

        self.surprise = 0
        self.td_error = 0
        # td moving average
        self.td_error_ma = 0

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, initial_context_message, initial_external_message):
        assert len(self.state_snapshot_stack) == 0
        self.cortical_column.reset(initial_context_message, initial_external_message)

    def sample_action(self):
        """Evaluate and sample actions."""
        action_values = self.evaluate_actions(with_planning=True)
        action_dist = self._get_action_selection_distribution(action_values, on_policy=True)
        action = self._rng.choice(self.n_actions, p=action_dist)
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

        current_state = self.cortical_column.layer.internal_forward_messages

        predicted_sr = self.predict_sr(current_state)
        generated_sr = self._generate_sr(
            self.sr_steps,
            initial_messages=current_state,
            initial_prediction=self.observation_messages,
            approximate_tail=self.approximate_tail,
        )
        self.td_update_sr(generated_sr, predicted_sr, current_state)
        return predicted_sr, generated_sr

    def reinforce(self, reward):
        """
        Adapt prior distribution of observations according to external reward.
            reward: float
        """
        # learn with mse loss
        lr, messages = self.observation_reward_lr, self.observation_messages

        self.observation_rewards += lr * messages * (reward - self.observation_rewards)

    def evaluate_actions(self, *, with_planning: bool = False):
        """Evaluate Q[s,a] for each action."""
        self._make_state_snapshot()

        n_actions = self.cortical_column.layer.external_input_size
        action_values = np.zeros(n_actions)
        dense_action = np.zeros_like(action_values)

        estimate_strategy = self._get_action_value_estimate_strategy(with_planning)

        for action in range(n_actions):
            # hacky way to clean previous one-hot, for 0-th does nothing
            dense_action[action-1] = 0
            # set current one-hot
            dense_action[action] = 1

            self.cortical_column.predict(
                context_messages=self.cortical_column.layer.context_messages,
                external_messages=dense_action
            )

            if estimate_strategy == ActionValueEstimate.PLAN:
                sr = self._generate_sr(
                    self.sr_steps,
                    initial_messages=self.cortical_column.layer.prediction_cells,
                    initial_prediction=self.cortical_column.layer.prediction_columns,
                    save_state=False
                )
            else:
                sr = self.predict_sr(self.cortical_column.layer.prediction_cells)

            # average value predicted by all variables
            action_values[action] = np.sum(
                sr * self.observation_rewards
            ) / self.cortical_column.layer.n_obs_vars

            self._restore_last_snapshot(pop=False)

        self.state_snapshot_stack.pop()
        return action_values

    def _generate_sr(
            self,
            n_steps,
            initial_messages,
            initial_prediction,
            approximate_tail=True,
            save_state=True
    ):
        if save_state:
            self._make_state_snapshot()

        sr = np.zeros_like(self.observation_messages)

        context_messages = initial_messages
        predicted_observation = initial_prediction

        t = -1
        for t in range(n_steps):
            sr += predicted_observation * self.gamma**t

            if self.sr_estimate_planning == SrEstimatePlanning.UNIFORM:
                action_dist = None
            else:
                # on/off-policy
                # NB: evaluate actions directly with prediction, not with n-step planning!
                action_values = self.evaluate_actions(with_planning=False)
                action_dist = self._get_action_selection_distribution(
                    action_values,
                    on_policy=self.sr_estimate_planning == SrEstimatePlanning.ON_POLICY
                )

            self.cortical_column.predict(context_messages, external_messages=action_dist)

            context_messages = self.cortical_column.layer.internal_forward_messages.copy()
            predicted_observation = self.cortical_column.layer.prediction_columns

        if approximate_tail:
            sr += self.predict_sr(context_messages) * self.gamma**(t+1)

        if save_state:
            self._restore_last_snapshot()

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

        self.td_error = np.mean(np.power(error_sr, 2))
        self.td_error_ma = lin_sum(self.td_error_ma, 0.2, self.td_error)

    def _get_action_selection_distribution(
            self, action_values, on_policy: bool = True
    ) -> np.ndarray:
        # off policy means greedy, on policy — with current exploration strategy
        if on_policy and self.exploration_policy == ExplorationPolicy.SOFTMAX:
            # normalize values before applying softmax to make the choice
            # of the softmax temperature scale invariant
            action_values = safe_divide(action_values, np.abs(action_values.sum()))
            action_dist = softmax(action_values, beta=self.inverse_temp)
        else:
            # greedy off policy or eps-greedy
            best_action = np.argmax(action_values)
            # make greedy policy
            # noinspection PyTypeChecker
            action_dist = sparse_to_dense([best_action], like=action_values)

            if on_policy and self.exploration_policy == ExplorationPolicy.EPS_GREEDY:
                # add uniform exploration
                action_dist[best_action] = 1 - self.exploration_eps
                action_dist[:] += self.exploration_eps / self.n_actions

        return action_dist

    def _get_action_value_estimate_strategy(self, with_planning: bool) -> ActionValueEstimate:
        if not with_planning or self.action_value_estimate == ActionValueEstimate.PREDICT:
            return ActionValueEstimate.PREDICT

        # with_planning == True
        if self.action_value_estimate == ActionValueEstimate.PLAN or self._should_plan():
            return ActionValueEstimate.PLAN

        return ActionValueEstimate.PREDICT

    def _should_plan(self):
        p = 1 - np.exp(-np.sqrt(self.td_error_ma) / .4)
        return self._rng.random() < p

    def _make_state_snapshot(self):
        self.state_snapshot_stack.append(
            self.cortical_column.make_state_snapshot()
        )

    def _restore_last_snapshot(self, pop: bool = True):
        snapshot = self.state_snapshot_stack.pop() if pop else self.state_snapshot_stack[-1]
        self.cortical_column.restore_last_snapshot(snapshot)

    @property
    def n_actions(self):
        return self.cortical_column.layer.external_input_size
