#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from enum import Enum, auto

from hima.common.sdr import sparse_to_dense
from hima.common.utils import softmax, safe_divide
from PIL import Image
import pygraphviz as pgv
import io

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
            new_cluster_weight,
            merge_threshold,
            update_period,
            gamma,
            reward_lr,
            inverse_temp,
            exploration_eps,
            trace_gamma,
            seed
    ):
        self.n_obs_states = n_obs_states
        self.n_actions = n_actions
        self.plan_steps = plan_steps
        self.new_cluster_weight = new_cluster_weight
        self.merge_threshold = merge_threshold
        self.update_period = update_period

        self.first_level_transitions = [dict() for _ in range(n_actions)]
        self.state_to_memory_trace = dict()
        self.state_to_sf = dict()
        self.second_level_transitions = [dict() for _ in range(n_actions)]
        self.cluster_to_states = dict()
        self.cluster_to_obs = dict()
        self.state_to_cluster = dict()
        self.obs_to_clusters = {obs: set() for obs in range(self.n_obs_states)}

        self.state = (-1, -1)
        self.cluster = {(-1, -1): 1.0}
        self.cluster_to_states[-1] = {(-1, -1)}
        self.state_to_cluster[(-1, -1)] = -1
        self.cluster_to_obs[-1] = -1

        self.gamma = gamma
        self.trace_gamma = trace_gamma
        self.reward_lr = reward_lr
        self.rewards = np.zeros(self.n_obs_states, dtype=np.float32)
        self.num_clones = np.zeros(self.n_obs_states, dtype=np.uint32)
        self.action_values = np.zeros(self.n_actions)
        self.memory_trace = np.zeros(self.n_obs_states)

        self.first_level_error = 0
        self.second_level_error = 0
        self.first_level_none = 0
        self.second_level_none = 0
        self.generalised = 0
        self.surprise = 0
        self.sf_steps = 0
        self.cluster_counter = 0
        self.time_step = 0
        self.goal_found = False
        self.learn = True
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.state_labels = dict()

        self.inverse_temp = inverse_temp
        if exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = exploration_eps

    def reset(self):
        self.state = (-1, -1)
        self.cluster = {(-1, -1): 1.0}
        self.goal_found = False
        self.surprise = 0
        self.first_level_error = 0
        self.second_level_error = 0
        self.generalised = 0
        self.first_level_none = 0
        self.second_level_none = 0
        self.sf_steps = 0
        self.action_values = np.zeros(self.n_actions)
        self.memory_trace = np.zeros(self.n_obs_states)

    def observe(self, observation, _reward, learn=True):
        # o_t, a_{t-1}
        obs_state, action = observation
        obs_dense = sparse_to_dense(obs_state, size=self.n_obs_states, dtype=np.float32)
        self.memory_trace = obs_dense + self.trace_gamma * self.memory_trace

        obs_state = int(obs_state[0])
        predicted_state = self.first_level_transitions[action].get(self.state)
        self.first_level_none = float(predicted_state is None)
        if (predicted_state is None) or (predicted_state[0] != obs_state):
            self.first_level_error = 1
            current_state = self._new_state(obs_state)
        else:
            self.first_level_error = 0
            current_state = predicted_state

        predicted_clusters = dict()
        obs_probs = np.zeros(self.n_obs_states)
        for c in self.cluster:
            pcs = self.second_level_transitions[action].get(c)
            if pcs is not None:
                for pc in pcs:
                    pc, prob = pc[:2], pc[-1]
                    prob = self.cluster[c] * prob
                    if pc in predicted_clusters:
                        predicted_clusters[pc] += prob
                    else:
                        predicted_clusters[pc] = prob
                    obs_probs[pc[0]] += prob

        self.second_level_none = obs_probs.sum()
        self.second_level_error = - np.log(obs_probs[obs_state] + EPS)
        # posterior update of predicted clusters (IMPORTANT! we didn't do that for previous tests)
        predicted_clusters = {c: p for c, p in predicted_clusters.items() if c[0] == obs_state}
        norm = sum(list(predicted_clusters.values())) + EPS
        predicted_clusters = {c: p/norm for c, p in predicted_clusters.items()}

        self.generalised = self.second_level_error < self.first_level_error

        # state induced cluster
        cluster = self.state_to_cluster.get(current_state)
        if cluster is not None:
            predicted_clusters = {(current_state[0], cluster): 1.0}

        current_clusters = predicted_clusters

        if learn:
            if current_state is not None:
                self.state_to_memory_trace[current_state] = self.memory_trace.copy()

            if (self.state is not None) and (current_state is not None):
                self.first_level_transitions[action][self.state] = current_state

            if cluster is None:
                # add state to a cluster with the most similar memory trace
                # or create a new cluster
                # p(c | s) = N * p(c) * exp(mem_s * mem_c)
                # where p(c) - chinese-restaurant prior
                candidates = list(self.obs_to_clusters[obs_state])

                traces = list()
                for c in candidates:
                    trace = [self.state_to_memory_trace[s] for s in self.cluster_to_states[c]]
                    trace = np.vstack(trace).mean(axis=0)
                    traces.append(trace)

                if len(traces) > 0:
                    traces = np.column_stack(traces)
                    # TODO add different similarity functions
                    norms = np.linalg.norm(traces, axis=0) * np.linalg.norm(self.memory_trace)
                    scores = (self.memory_trace @ traces) / (norms + EPS)
                else:
                    scores = np.empty(0, dtype=np.float32)
                scores = np.append(scores, self.merge_threshold)
                scores = np.exp(scores)

                lengths = [
                    len(self.cluster_to_states[c])
                    for c in candidates
                ]
                lengths.append(self.new_cluster_weight)
                lengths = np.array(lengths, dtype=np.float32)

                c_prior = lengths / (lengths.sum() + EPS)
                c_posterior = c_prior * scores
                c_posterior = c_posterior / (c_posterior.sum() + EPS)

                candidates.append(-1)
                winner = self._rng.choice(candidates, p=c_posterior)
                if winner == -1:
                    winner = self.cluster_counter
                    self.cluster_counter += 1
                    self.cluster_to_states[winner] = set()
                    self.cluster_to_obs[winner] = obs_state
                    self.obs_to_clusters[obs_state].add(winner)

                self.cluster_to_states[winner].add(current_state)
                self.state_to_cluster[current_state] = winner

            if (self.time_step % self.update_period) == 0:
                self._update_second_level()

        self.state = current_state
        self.cluster = current_clusters
        self.time_step += 1

    def sample_action(self):
        action_values = self.evaluate_actions()
        action_dist = self._get_action_selection_distribution(
            action_values, on_policy=True
        )
        action = self._rng.choice(self.n_actions, p=action_dist)
        return action

    def reinforce(self, reward):
        if self.state is not None:
            obs = self.state[0]
            self.rewards[obs] += self.reward_lr * (
                    reward -
                    self.rewards[obs]
            )

    def evaluate_actions(self):
        self.action_values = np.zeros(self.n_actions)

        planning_steps = 0
        self.goal_found = False
        for action in range(self.n_actions):
            predicted_state = self.first_level_transitions[action].get(self.state)
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
                *[self._predict(predicted_states, a) for a in range(self.n_actions)]
            )
            obs_states = np.array(
                self._convert_to_obs_states(predicted_states),
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

    def sleep_phase(self, iterations):
        ...

    def _update_second_level(self):
        for cluster_id in self.cluster_to_states:
            # update transition matrix
            for a, d_a in enumerate(self.second_level_transitions):
                predicted_states = [
                    self.first_level_transitions[a][s]
                    for s in self.cluster_to_states[cluster_id]
                    if s in self.first_level_transitions[a]
                ]
                predicted_clusters = [
                    self.state_to_cluster[s] for s in predicted_states
                    if s in self.state_to_cluster
                ]
                predicted_clusters, counts = np.unique(predicted_clusters, return_counts=True)
                # assert len(set([self.cluster_to_obs[c] for c in predicted_clusters])) <= 1
                if len(counts) > 0:
                    probs = counts / counts.sum()
                    cluster = (self.cluster_to_obs[cluster_id], cluster_id)
                    pred_clusters = {(self.cluster_to_obs[pc], pc, p) for pc, p in zip(predicted_clusters, probs)}
                    d_a[cluster] = pred_clusters

    def _update_cluster(
            self,
            new_states: set,
            cluster_id: int,
            obs_state: int,
            old_cluster_assignment: dict=None
    ):
        """
            Update cluster state diff
            Assign new_states to another cluster and also update clusters that are affected.

            old_cluster_assignment: dict
        """
        for s in new_states:
            if s in self.state_to_cluster:
                # remove from old cluster
                old_c = self.state_to_cluster[s]
                if old_c != cluster_id:
                    self.cluster_to_states[old_c].remove(s)
            self.state_to_cluster[s] = cluster_id

        if cluster_id in self.cluster_to_states:
            removed_states = self.cluster_to_states[cluster_id].difference(new_states)
            for s in removed_states:
                old_cluster = None
                if old_cluster_assignment is not None:
                    old_cluster = old_cluster_assignment.get(s)
                if (old_cluster is not None) and (old_cluster != cluster_id):
                    self.state_to_cluster[s] = old_cluster
                    self.cluster_to_states[old_cluster].add(s)
                else:
                    if s in self.state_to_cluster:
                        self.state_to_cluster.pop(s)

        self.cluster_to_states[cluster_id] = new_states
        self.cluster_to_obs[cluster_id] = obs_state
        self.obs_to_clusters[obs_state].add(cluster_id)

    def _destroy_cluster(self, cluster: tuple):
        obs_state, cluster_id = cluster
        if cluster_id in self.cluster_to_states:
            states = self.cluster_to_states.pop(cluster_id)
            for s in states:
                if s in self.state_to_cluster:
                    self.state_to_cluster.pop(s)

        if obs_state in self.obs_to_clusters:
            self.obs_to_clusters[obs_state].remove(cluster_id)

        for d_a in self.second_level_transitions:
            if cluster in d_a:
                d_a.pop(cluster)

    def _predict(self, state: set, action: int, expand_clusters: bool = False) -> set:
        state_expanded = state
        if expand_clusters:
            clusters = [self.state_to_cluster.get(s) for s in state]
            state_expanded = state_expanded.union(
                *[self.cluster_to_states[c] for c in clusters if c is not None]
            )

        d_a = self.first_level_transitions[action]
        predicted_state = set()
        for s in state_expanded:
            if s in d_a:
                predicted_state.add(d_a[s])
        return predicted_state

    def _convert_to_obs_states(self, states):
        return [s[0] for s in states if s is not None]

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
    def num_clusters(self):
        return len(self.cluster_to_states)

    @property
    def average_cluster_size(self):
        return np.array([len(self.cluster_to_states[c]) for c in self.cluster_to_states]).mean()

    @property
    def num_transitions_second_level(self):
        return sum([len(d_a) for d_a in self.second_level_transitions])

    @property
    def draw_transition_graph(self, threshold=0.1):
        g = pgv.AGraph(strict=False, directed=True)

        for a, d_a in enumerate(self.second_level_transitions):
            for c in d_a:
                outs = d_a[c]
                for out in outs:
                    if out[-1] > threshold:
                        g.add_edge(c, out[:2], label=f"{a}:{round(out[-1], 2)}")

        g.layout(prog='dot')
        buf = io.BytesIO()
        g.draw(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        return im