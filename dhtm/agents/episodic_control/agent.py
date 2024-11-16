#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from enum import Enum, auto

from dhtm.common.sdr import sparse_to_dense
from dhtm.common.utils import softmax, safe_divide

EPS = 1e-24


class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()


class ECAgent:
    def __init__(
            self,
            n_obs_states,
            n_actions,
            plan_steps,
            gamma,
            reward_lr,
            inverse_temp,
            exploration_eps,
            seed
    ):
        self.n_obs_states = n_obs_states
        self.n_actions = n_actions
        self.plan_steps = plan_steps
        self.dictionaries = [dict() for _ in range(n_actions)]
        self.state = (0, 0)
        self.gamma = gamma
        self.reward_lr = reward_lr
        self.rewards = np.zeros(self.n_obs_states, dtype=np.float32)
        self.num_clones = np.zeros(self.n_obs_states, dtype=np.uint32)
        self.action_values = np.zeros(self.n_actions)
        self.surprise = 0
        self.sf_steps = 0
        self.goal_found = False
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        self.inverse_temp = inverse_temp
        if exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = exploration_eps

    def reset(self):
        self.state = (0, 0)
        self.goal_found = False
        self.surprise = 0
        self.sf_steps = 0
        self.action_values = np.zeros(self.n_actions)

    def observe(self, observation, _reward, learn=True):
        # o_t, a_{t-1}
        obs_state, action = observation
        obs_state = int(obs_state[0])

        predicted_state = self.dictionaries[action].get(self.state)
        if (predicted_state is None) or (predicted_state[0] != obs_state):
            current_state = self._new_state(obs_state)
        else:
            current_state = predicted_state

        if learn:
            self.dictionaries[action][self.state] = current_state

        self.state = current_state

    def sample_action(self):
        action_values = self.evaluate_actions()
        action_dist = self._get_action_selection_distribution(
            action_values, on_policy=True
        )
        action = self._rng.choice(self.n_actions, p=action_dist)
        return action

    def reinforce(self, reward):
        self.rewards[self.state[0]] += self.reward_lr * (
                reward -
                self.rewards[self.state[0]]
        )

    def evaluate_actions(self):
        self.action_values = np.zeros(self.n_actions)

        planning_steps = 0
        self.goal_found = False
        for action in range(self.n_actions):
            predicted_state = self.dictionaries[action].get(self.state)
            sf, steps, gf = self.generate_sf(predicted_state)
            self.goal_found = gf or self.goal_found
            planning_steps += steps
            self.action_values[action] = np.sum(sf * self.rewards)

        self.sf_steps = planning_steps / self.n_actions
        return self.action_values

    def generate_sf(self, initial_state):
        sf = np.zeros(self.n_obs_states, dtype=np.float32)
        goal_found = False

        if initial_state is None:
            return sf, 0, False

        sf[initial_state[0]] = 1

        predicted_states = {initial_state}

        discount = self.gamma

        i = -1
        for i in range(self.plan_steps):
            # uniform strategy
            predicted_states = set().union(
                *[{d.get(state) for d in self.dictionaries} for state in predicted_states]
            )
            obs_states = np.array(
                [s[0] for s in predicted_states if s is not None],
                dtype=np.uint32
            )
            obs_states, counts = np.unique(obs_states, return_counts=True)
            obs_probs = np.zeros_like(sf)
            obs_probs[obs_states] = counts
            obs_probs /= (obs_probs.sum() + EPS)
            sf += discount * obs_probs

            if len(obs_states) == 0:
                break

            if np.any(
                self.rewards[list(obs_states)] > 0
            ):
                goal_found = True
                break

            discount *= self.gamma

        return sf, i+1, goal_found

    def _new_state(self, obs_state):
        h = int(self.num_clones[obs_state])
        self.num_clones[obs_state] += 1

        return obs_state, h

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

    @property
    def total_num_clones(self):
        return self.num_clones.sum()

    @property
    def dict_size(self):
        return sum([len(d) for d in self.dictionaries])
