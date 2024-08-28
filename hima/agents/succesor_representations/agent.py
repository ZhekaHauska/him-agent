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
from hima.common.utils import softmax, safe_divide
from hima.common.smooth_values import SSValue
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from copy import copy
from hima.modules.belief.utils import EPS


class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()


class SrEstimatePlanning(Enum):
    UNIFORM = 1
    ON_POLICY = auto()
    OFF_POLICY = auto()


class BioHIMA:
    def __init__(
            self,
            cortical_column: CorticalColumn,
            *,
            gamma: float = 0.99,
            reward_lr: float = 0.01,
            learn_rewards_from_state: bool = True,
            plan_steps: int = 1,
            inverse_temp: float = 1.0,
            exploration_eps: float = -1,
            sr_estimate_planning: str = 'uniform',
            sr_early_stop_uniform: float | None = None,
            sr_early_stop_goal: float | None = None,
            sr_early_stop_surprise: float | None = None,
            lr_surprise=(0.2, 0.01),
            seed: int | None,
    ):
        self.reward_lr = reward_lr
        self.learn_rewards_from_state = learn_rewards_from_state
        self.cortical_column = cortical_column
        self.gamma = gamma
        self.max_plan_steps = plan_steps
        self.inverse_temp = inverse_temp

        if exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = exploration_eps

        self.sr_early_stop_uniform = sr_early_stop_uniform
        self.sr_early_stop_goal = sr_early_stop_goal
        self.sr_early_stop_surprise = sr_early_stop_surprise

        self.sr_estimate_planning = SrEstimatePlanning[sr_estimate_planning.upper()]

        layer = self.cortical_column.layer
        if self.learn_rewards_from_state:
            rewards = np.zeros((layer.n_hidden_vars, layer.n_hidden_states))
        else:
            rewards = np.zeros((layer.n_obs_vars, layer.n_obs_states))
        self.rewards = rewards.flatten()

        self.observation_messages = np.zeros_like(layer.observation_messages)

        self.state_snapshot_stack = deque()

        self.sf = None
        self.action_values = None
        self.action_dist = None
        self.action = None

        # metrics
        self.ss_surprise = SSValue(*lr_surprise)
        self.sf_steps = 0

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self):
        assert len(self.state_snapshot_stack) == 0

        self.sf = None
        self.action_values = None
        self.action_dist = None
        self.action = None
        self.sf_steps = 0

        self.cortical_column.reset()

    def sample_action(self):
        """Evaluate and sample actions."""
        self.action_values = self.evaluate_actions()
        self.action_dist = self._get_action_selection_distribution(
            self.action_values, on_policy=True
        )
        self.action = self._rng.choice(self.n_actions, p=self.action_dist)
        return self.action

    def observe(self, observation, learn=True):
        """
        Main learning routine
            observation: tuple (image, action)
                events: sparse sdr
                action: sparse sdr
        """
        events, action = observation

        if events is not None:
            # predict current events using observed action
            self.cortical_column.observe(events, action, learn=learn)
            encoded_obs = self.cortical_column.output_sdr.sparse
            self.observation_messages = sparse_to_dense(encoded_obs, like=self.observation_messages)
        else:
            return

        if not learn:
            return

        self.ss_surprise.update(self.cortical_column.surprise)

        return self.sf

    def reinforce(self, reward):
        if self.learn_rewards_from_state:
            messages = self.cortical_column.layer.internal_messages
        else:
            messages = self.cortical_column.layer.observation_messages

        deltas = messages * (reward - self.rewards)

        self.rewards += self.reward_lr * deltas

    def evaluate_actions(self):
        """Evaluate Q[s,a] for each action."""
        self._make_state_snapshot()

        n_actions = self.cortical_column.layer.external_input_size
        action_values = np.zeros(n_actions)
        dense_action = np.zeros_like(action_values)

        for action in range(n_actions):
            # hacky way to clean previous one-hot, for 0-th does nothing
            dense_action[action - 1] = 0
            # set current one-hot
            dense_action[action] = 1

            self.cortical_column.predict(
                context_messages=self.cortical_column.layer.context_messages,
                external_messages=dense_action
            )

            if self.learn_rewards_from_state:
                sf, self.sf_steps, sr = self.generate_sf(
                    self.max_plan_steps,
                    initial_messages=self.cortical_column.layer.prediction_cells,
                    initial_prediction=self.cortical_column.layer.prediction_columns,
                    return_sr=True,
                    save_state=False,
                )

                action_values[action] = np.sum(
                    sr * self.rewards
                ) / self.cortical_column.layer.n_hidden_vars
            else:
                sf, self.sf_steps = self.generate_sf(
                    self.max_plan_steps,
                    initial_messages=self.cortical_column.layer.prediction_cells,
                    initial_prediction=self.cortical_column.layer.prediction_columns,
                    save_state=False,
                )

                action_values[action] = np.sum(
                    sf * self.rewards
                ) / self.cortical_column.layer.n_obs_vars

            self._restore_last_snapshot(pop=False)

        self.state_snapshot_stack.pop()
        return action_values

    def generate_sf(
            self,
            n_steps,
            initial_messages,
            initial_prediction,
            save_state=True,
            return_predictions=False,
            return_sr=False
    ):
        """
            n_steps: number of prediction steps. If n_steps is 0 and approximate_tail is True,
            then this function is equivalent to predict_sr.
        """
        predictions = []
        sr = np.zeros_like(initial_messages)

        if save_state:
            self._make_state_snapshot()

        sf = np.zeros_like(self.observation_messages)

        context_messages = initial_messages
        predicted_observation = initial_prediction

        discount = 1.0
        t = -1
        for t in range(n_steps):
            if self.learn_rewards_from_state:
                early_stop = self._early_stop_planning(
                    context_messages.reshape(
                        self.cortical_column.layer.n_hidden_vars, -1
                    )
                )
            else:
                early_stop = self._early_stop_planning(
                    predicted_observation.reshape(
                        self.cortical_column.layer.n_obs_vars, -1
                    )
                )

            sf += predicted_observation * discount

            if return_sr:
                sr += context_messages * discount

            if self.sr_estimate_planning == SrEstimatePlanning.UNIFORM:
                action_dist = None
            else:
                # on/off-policy
                action_values = self.evaluate_actions()
                action_dist = self._get_action_selection_distribution(
                    action_values,
                    on_policy=self.sr_estimate_planning == SrEstimatePlanning.ON_POLICY
                )

            self.cortical_column.predict(context_messages, external_messages=action_dist)

            context_messages = self.cortical_column.layer.internal_messages.copy()
            predicted_observation = self.cortical_column.layer.prediction_columns

            discount *= self.gamma

            if return_predictions:
                predictions.append(copy(predicted_observation))

            if np.allclose(predicted_observation, 0):
                break

            if early_stop:
                break

        if save_state:
            self._restore_last_snapshot()

        output = [sf, t+1]

        if return_predictions:
            output.append(predictions)
        if return_sr:
            output.append(sr)

        return output

    @property
    def current_state(self):
        return self.cortical_column.layer.internal_messages

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

    def _make_state_snapshot(self):
        self.state_snapshot_stack.append(
            self.cortical_column.make_state_snapshot()
        )

    def _restore_last_snapshot(self, pop: bool = True):
        snapshot = self.state_snapshot_stack.pop() if pop else self.state_snapshot_stack[-1]
        self.cortical_column.restore_last_snapshot(snapshot)

    def _early_stop_planning(self, messages: np.ndarray) -> bool:
        n_vars, n_states = messages.shape

        if self.sr_early_stop_uniform is not None:
            uni_dkl = (
                    np.log(n_states) +
                    np.sum(
                        messages * np.log(
                            np.clip(
                                messages, EPS, None
                            )
                        ),
                        axis=-1
                    )
            )

            uniform = uni_dkl.mean() < self.sr_early_stop_uniform
        else:
            uniform = False

        if self.sr_early_stop_goal is not None:
            goal = np.any(
                np.sum(
                    (messages.flatten() * (self.rewards > 0)).reshape(
                        n_vars, -1
                    ),
                    axis=-1
                ) > self.sr_early_stop_goal
            )
        else:
            goal = False

        if self.sr_early_stop_surprise is not None:
            surprise = self.ss_surprise.mean > self.sr_early_stop_surprise
        else:
            surprise = False

        return uniform or goal or surprise

    @property
    def n_actions(self):
        return self.cortical_column.layer.external_input_size

    @property
    def surprise(self):
        return self.ss_surprise.current_value

