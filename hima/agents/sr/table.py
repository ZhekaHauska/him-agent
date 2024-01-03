#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from enum import Enum, auto
from typing import Literal
from hima.common.utils import softmax


class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()


class SrEstimate(Enum):
    UNIFORM = 1
    ON_POLICY = auto()
    OFF_POLICY = auto()


class SRAgent:
    def __init__(
            self,
            n_states,
            n_actions,
            gamma,
            sr_lr,
            rew_lr,
            exploration_eps=-1,
            inverse_temp=1.0,
            sr_estimate: Literal['uniform', 'on_policy', 'off_policy'] = 'on_policy',
            seed=None
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.inverse_temp = inverse_temp
        self.sr_lr = sr_lr
        self.rew_lr = rew_lr
        self.seed = seed

        if exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = exploration_eps

        self.sr_estimate = SrEstimate[sr_estimate.upper()]

        self.sr = np.zeros((n_actions, n_states, n_states))
        self.rewards = np.zeros(n_states)

        self.previous_state = None
        self.current_state = None
        self.previous_action = None
        self.current_action = None
        self.td_error = 0

        self._rnd = np.random.default_rng(self.seed)

    def observe(self, state, action, learn=True):
        """
            observe s_t and a_t
        """
        self.previous_state = self.current_state
        self.previous_action = self.current_action

        self.current_state = state
        self.current_action = action

        if learn:
            self.td_update()

    def act(self, state):
        if self.exploration_policy == ExplorationPolicy.EPS_GREEDY:
            if self._rnd.random() < self.exploration_eps:
                return self._rnd.integers(0, self.n_actions)
            else:
                return np.argmax(self.get_action_values(state))
        else:
            self._rnd.choice(
                np.arange(self.n_actions),
                p=softmax(self.get_action_values(state), beta=self.inverse_temp)
            )

    def reinforce(self, reward):
        assert self.current_state is not None
        self.rewards[self.current_state] += self.rew_lr * (
                reward - self.rewards[self.current_state]
        )

    def reset(self):
        self.previous_state = None
        self.current_state = None
        self.previous_action = None
        self.current_action = None
        self.td_error = 0

    def td_update(self):
        if self.previous_action is None or self.previous_state is None:
            return

        predicted_sr = self.sr[self.previous_action, self.previous_state]

        dense_state = np.zeros(self.n_states)
        dense_state[self.current_state] = 1

        if self.sr_estimate == SrEstimate.ON_POLICY:
            if self.current_action is not None:
                target_sr = dense_state + self.gamma * (
                    self.sr[self.current_action, self.current_state]
                )
            else:
                target_sr = dense_state + self.gamma * (
                    np.mean(self.sr[:, self.current_state], axis=0)
                )
        elif self.sr_estimate == SrEstimate.UNIFORM:
            target_sr = dense_state + self.gamma * (
                np.mean(self.sr[:, self.current_state], axis=0)
            )
        else:
            best_action = np.argmax(self.get_action_values(self.current_state))
            target_sr = dense_state + self.gamma * (
                self.sr[best_action, self.current_state]
            )

        td_error = target_sr - predicted_sr

        self.sr[self.previous_action, self.previous_state] += self.sr_lr * td_error

        self.td_error = np.sum(np.power(td_error, 2))

    def get_action_values(self, state):
        assert state is not None
        return np.dot(self.sr[:, state], self.rewards)
