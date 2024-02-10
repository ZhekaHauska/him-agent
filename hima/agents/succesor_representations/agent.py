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
from hima.modules.belief.utils import normalize
from hima.common.smooth_values import SSValue
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from hima.modules.baselines.srtd import SRTD
from hima.agents.succesor_representations.striatum import Striatum
from hima.modules.baselines.lstm import to_numpy, TLstmLayerHiddenState
from copy import copy
import torch
from hima.modules.belief.utils import EPS


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
            plan_steps: int = 1,
            use_cached_plan: bool = False,
            learn_cached_plan: bool = False,
            td_steps: int = 1,
            approximate_tail: bool = True,
            inverse_temp: float = 1.0,
            exploration_eps: float = -1,
            action_value_estimate: str = 'plan',
            sr_estimate_planning: str = 'uniform',
            sr_early_stop_uniform: float | None = None,
            sr_early_stop_goal: float | None = None,
            sr_early_stop_surprise: float | None = None,
            lr_surprise=(0.2, 0.01),
            lr_td_error=(0.2, 0.01),
            adaptive_sr: bool = True,
            adaptive_lr: bool = True,
            srtd: SRTD | None,
            pattern_memory: Striatum | None,
            use_sf_as_state: bool = False,
            seed: int | None,
    ):
        self.observation_reward_lr = observation_reward_lr
        self.max_striatum_lr = striatum_lr
        self.cortical_column = cortical_column
        self.srtd = srtd
        self.pattern_memory = pattern_memory
        self.use_sf_as_state = use_sf_as_state
        self.gamma = gamma
        self.max_plan_steps = plan_steps
        self.use_cached_plan = use_cached_plan
        self.learn_cached_plan = learn_cached_plan
        self.td_steps = td_steps
        self.approximate_tail = approximate_tail
        self.inverse_temp = inverse_temp
        self.adaptive_sr = adaptive_sr
        self.adaptive_lr = adaptive_lr

        if exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = exploration_eps

        self.sr_early_stop_uniform = sr_early_stop_uniform
        self.sr_early_stop_goal = sr_early_stop_goal
        self.sr_early_stop_surprise = sr_early_stop_surprise

        self._action_value_estimate = ActionValueEstimate[action_value_estimate.upper()]
        self.sr_estimate_planning = SrEstimatePlanning[sr_estimate_planning.upper()]

        layer = self.cortical_column.layer
        layer_obs_size = layer.n_obs_states * layer.n_obs_vars
        layer_hidden_size = layer.n_hidden_states * layer.n_hidden_vars

        observation_rewards = np.zeros((layer.n_obs_vars, layer.n_obs_states))
        self.observation_rewards = observation_rewards.flatten()
        self.observation_messages = np.zeros_like(self.observation_rewards)

        # state backups for model-free TD
        self.previous_state = self.cortical_column.layer.internal_forward_messages.copy()
        self.previous_observation = np.zeros_like(self.observation_rewards)

        # TODO move it to Striatum class
        if self.use_sf_as_state:
            self.striatum_weights = np.zeros((layer_obs_size, layer_obs_size))
        else:
            self.striatum_weights = np.zeros((layer_hidden_size, layer_obs_size))

        self.state_snapshot_stack = deque()

        self.predicted_sf = None
        self.generated_sf = None
        self.action_values = None
        self.action_dist = None
        self.action = None

        # metrics
        self.ss_td_error = SSValue(*lr_td_error)
        self.ss_surprise = SSValue(*lr_surprise)
        self.sf_steps = 0

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, initial_context_message, initial_external_message):
        assert len(self.state_snapshot_stack) == 0

        self.predicted_sf = None
        self.generated_sf = None
        self.action_values = None
        self.action_dist = None
        self.action = None
        self.sf_steps = 0

        self.cortical_column.reset(initial_context_message, initial_external_message)

        self.previous_state = self.cortical_column.layer.internal_forward_messages.copy()
        self.previous_observation = np.zeros_like(self.observation_rewards)

    def sample_action(self):
        """Evaluate and sample actions."""
        self.action_values = self.evaluate_actions(with_planning=True)
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

        # striatum TD learning
        if self.td_steps > 0:
            self.predicted_sf, self.generated_sf, td_error = self.td_update_sf()
            self.ss_td_error.update(td_error)

        if self.plan_steps > 0 and self.pattern_memory is not None and self.learn_cached_plan:
            self.update_planned_sf()

        self.ss_surprise.update(self.cortical_column.surprise)

        return self.predicted_sf, self.generated_sf

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
            dense_action[action - 1] = 0
            # set current one-hot
            dense_action[action] = 1

            self.cortical_column.predict(
                context_messages=self.cortical_column.layer.context_messages,
                external_messages=dense_action
            )

            if estimate_strategy == ActionValueEstimate.PLAN:
                if self.use_cached_plan:
                    sf = self.predict_sf(self.cortical_column.layer.prediction_cells, area=1)
                else:
                    sf, self.sf_steps = self.generate_sf(
                        self.plan_steps,
                        initial_messages=self.cortical_column.layer.prediction_cells,
                        initial_prediction=self.cortical_column.layer.prediction_columns,
                        approximate_tail=self.approximate_tail,
                        save_state=False,
                    )
            else:
                sf = self.predict_sf(self.cortical_column.layer.prediction_cells)

            # average value predicted by all variables
            action_values[action] = np.sum(
                sf * self.observation_rewards
            ) / self.cortical_column.layer.n_obs_vars

            self._restore_last_snapshot(pop=False)

        self.state_snapshot_stack.pop()
        return action_values

    def generate_sf(
            self,
            n_steps,
            initial_messages,
            initial_prediction,
            approximate_tail=True,
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
            early_stop = self._early_stop_planning(predicted_observation)

            sf += predicted_observation * discount

            if return_sr:
                sr += context_messages * discount

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

            discount *= self.gamma

            if return_predictions:
                predictions.append(copy(predicted_observation))

            if early_stop:
                break

        if approximate_tail:
            sf += self.predict_sf(context_messages) * discount

        if save_state:
            self._restore_last_snapshot()

        output = [sf, t+1]

        if return_predictions:
            output.append(predictions)
        if return_sr:
            output.append(sr)

        return output

    def predict_sf(self, hidden_vars_dist, area=0):
        if self.srtd is not None:
            msg = torch.tensor(hidden_vars_dist).float().to(self.srtd.device)
            sr = to_numpy(self.srtd.predict_sr(msg, target=True))
        elif self.pattern_memory is not None:
            if self.use_sf_as_state and area == 0:
                sr = self.pattern_memory.predict(hidden_vars_dist, area=1, learn=False)
                sr = sr.reshape(self.cortical_column.layer.n_obs_vars, -1)
                sr = np.dot(normalize(sr).flatten(), self.striatum_weights)
                sr /= self.cortical_column.layer.n_obs_vars
            else:
                sr = self.pattern_memory.predict(hidden_vars_dist, area=area, learn=False)
                sr.reshape(self.cortical_column.layer.n_obs_vars, -1)
        else:
            sr = np.dot(hidden_vars_dist, self.striatum_weights)
            sr /= self.cortical_column.layer.n_hidden_vars
        return sr

    def td_update_sf(self):
        target_sf, _ = self.generate_sf(
            self.td_steps,
            initial_messages=self.cortical_column.layer.internal_forward_messages,
            initial_prediction=self.observation_messages,
            approximate_tail=True,
        )

        if self.srtd is not None:
            current_state = torch.tensor(self.current_state).float().to(self.srtd.device)
            predicted_sf = self.srtd.predict_sr(current_state, target=False)
            target_sf = torch.tensor(target_sf)
            target_sf = target_sf.float().to(self.srtd.device)

            td_error = self.srtd.compute_td_loss(
                target_sf,
                predicted_sf
            )
            predicted_sf = to_numpy(predicted_sf)
            target_sf = to_numpy(target_sf)
        elif self.pattern_memory is not None:
            if self.use_sf_as_state:
                sr = self.pattern_memory.predict(self.current_state, area=1, learn=False)
                sr = sr.reshape(self.cortical_column.layer.n_obs_vars, -1)

                predicted_sf = self.predict_sf(self.current_state)
                prediction_cells = normalize(sr).flatten()
                error_sr = target_sf - predicted_sf

                # dSR / dW for linear model
                delta_w = np.outer(prediction_cells, error_sr)

                self.striatum_weights += self.striatum_lr * delta_w
                self.striatum_weights = np.clip(self.striatum_weights, 0, None)

                td_error = np.mean(np.power(error_sr, 2))
            else:
                predicted_sf = self.pattern_memory.predict(self.current_state, learn=True)
                td_error = self.pattern_memory.update_weights(target_sf)
        else:
            predicted_sf = self.predict_sf(self.current_state)
            prediction_cells = self.current_state
            error_sr = target_sf - predicted_sf

            # dSR / dW for linear model
            delta_w = np.outer(prediction_cells, error_sr)

            self.striatum_weights += self.striatum_lr * delta_w
            self.striatum_weights = np.clip(self.striatum_weights, 0, None)

            td_error = np.mean(np.power(error_sr, 2))

        return predicted_sf, target_sf, td_error

    def update_planned_sf(self):
        target_sf, self.sf_steps = self.generate_sf(
            self.plan_steps,
            initial_messages=self.cortical_column.layer.internal_forward_messages,
            initial_prediction=self.observation_messages,
            approximate_tail=False,
        )
        self.pattern_memory.predict(self.current_state, area=1, learn=True)
        self.pattern_memory.update_weights(target_sf, area=1)

    @property
    def current_state(self):
        return self.cortical_column.layer.internal_forward_messages

    @property
    def action_value_estimate(self):
        return self._action_value_estimate

    @action_value_estimate.setter
    def action_value_estimate(self, value: str):
        self._action_value_estimate = ActionValueEstimate[value.upper()]

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
        p = np.clip(self.ss_td_error.norm_value, 0, 1)
        return self._rng.random() < p

    def _make_state_snapshot(self):
        self.state_snapshot_stack.append(
            self.cortical_column.make_state_snapshot()
        )

    def _restore_last_snapshot(self, pop: bool = True):
        snapshot = self.state_snapshot_stack.pop() if pop else self.state_snapshot_stack[-1]
        self.cortical_column.restore_last_snapshot(snapshot)

    def _early_stop_planning(self, predicted_observation: np.ndarray) -> bool:
        if self.sr_early_stop_uniform is not None:
            uni_dkl = (
                    np.log(self.cortical_column.layer.n_obs_states) +
                    np.sum(
                        predicted_observation * np.log(
                            np.clip(
                                predicted_observation, EPS, None
                            )
                        )
                    )
            )

            uniform = uni_dkl < self.sr_early_stop_uniform
        else:
            uniform = False

        if self.sr_early_stop_goal is not None:
            goal = (
                np.sum(predicted_observation[self.observation_rewards > 0]) >
                self.sr_early_stop_goal
            )
        else:
            goal = False

        if self.sr_early_stop_surprise is not None:
            surprise = self.ss_surprise.mean > self.sr_early_stop_surprise
        else:
            surprise = False

        return uniform or goal or surprise

    @property
    def striatum_lr(self):
        if self.adaptive_lr:
            lr = self.max_striatum_lr * (1 - np.clip(self.ss_surprise.norm_value, 0, 1))
        else:
            lr = self.max_striatum_lr
        return lr

    @property
    def relative_log_surprise(self):
        s_base_log = np.log(np.log(self.cortical_column.layer.n_obs_states))
        s_current = np.clip(self.ss_surprise.current_value, 1e-7, None)
        s_min = np.clip(self.ss_surprise.mean - self.ss_surprise.std, 1e-7, None)
        return (1 - np.log(s_current) / s_base_log) / (1 - np.log(s_min) / s_base_log)

    @property
    def plan_steps(self):
        if self.adaptive_sr:
            sr_steps = int(
                max(
                    1,
                    np.round(
                        self.max_plan_steps *
                        np.clip(self.ss_td_error.norm_value, 0, 1) *
                        self.relative_log_surprise
                    )
                )
            )
        else:
            sr_steps = self.max_plan_steps
        return sr_steps

    @property
    def n_actions(self):
        return self.cortical_column.layer.external_input_size

    @property
    def surprise(self):
        return self.ss_surprise.current_value

    @property
    def td_error(self):
        return self.ss_td_error.current_value


class LstmBioHima(BioHIMA):
    """Patch-like adaptation of BioHIMA to work with LSTM layer."""

    def __init__(
            self, cortical_column: CorticalColumn,
            **kwargs
    ):
        super().__init__(cortical_column, **kwargs)

    # noinspection PyMethodOverriding
    def generate_sf(
            self,
            n_steps,
            initial_messages,
            initial_prediction,
            approximate_tail=True,
            save_state=True,
            return_predictions=False,
    ):
        """
            n_steps: number of prediction steps. If n_steps is 0 and approximate_tail is True,
            then this function is equivalent to predict_sr.
        """
        predictions = []

        if save_state:
            self._make_state_snapshot()

        sr = np.zeros_like(self.observation_messages)

        # represent state after getting observation
        context_messages = initial_messages
        # represent predicted observation
        predicted_observation = initial_prediction

        discount = 1.0
        t = -1
        for t in range(n_steps):
            early_stop = self._early_stop_planning(predicted_observation)

            sr += predicted_observation * discount

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

            # THE ONLY ADDED CHANGE TO HIMA: explicitly observe predicted_observation
            self.cortical_column.layer.observe(predicted_observation, learn=False)
            # setting context is needed for action evaluation further on
            self.cortical_column.layer.set_context_messages(
                self.cortical_column.layer.internal_forward_messages
            )
            # ======

            discount *= self.gamma

            if return_predictions:
                predictions.append(copy(predicted_observation))

            if early_stop:
                break

        if approximate_tail:
            sr += self.predict_sf(context_messages) * discount

        if save_state:
            self._restore_last_snapshot()

        # sr /= self.cortical_column.layer.n_hidden_vars

        if return_predictions:
            return sr, t+1, predictions
        else:
            return sr, t+1

    def _extract_state_from_context(self, context_messages: TLstmLayerHiddenState):
        # extract model state from layer state
        state_out, _ = context_messages[1]

        # convert model hidden state to probabilities
        state_probs_out = self.cortical_column.layer.model.to_probabilistic_out_state(state_out)
        return state_probs_out.detach()

    def predict_sf(self, context_messages: TLstmLayerHiddenState):
        msg = to_numpy(self._extract_state_from_context(context_messages))
        return super().predict_sf(msg)

    @property
    def current_state(self):
        return to_numpy(self._extract_state_from_context(
            self.cortical_column.layer.internal_forward_messages
        ))
