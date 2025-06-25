#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle

import numpy as np
from enum import Enum, auto

import wandb

from hima.common.sdr import sparse_to_dense
from hima.common.smooth_values import SSValue
from hima.common.utils import softmax, safe_divide
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import entropy
from PIL import Image
import pygraphviz as pgv
import io

from hima.modules.belief.utils import normalize

EPS = 1e-24

def norm_cdf(z):
    """
        Normal CDF approximation https://doi.org/10.1016/0096-3003(95)00190-5
        works for z in [-8, 8]
        x: standard normal random variable, i.e. z = (x - mean)/std
    """
    b1, b2, b3 = -0.0004406, 0.0418198, 0.9
    pol = b1 * np.power(z, 5) + b2 * np.power(z, 3) + b3 * z
    return 1/(1 + np.exp(-np.sqrt(np.pi) * pol))

def euclidian_sim(X, Y=None):
    return np.exp(-euclidean_distances(X, Y))

def dummy_sim(X, Y=None):
    if Y is None:
        return np.zeros((X.shape[0], X.shape[0]))
    else:
        return np.zeros((X.shape[0], Y.shape[0]))

class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()


class ECAgent:
    similarities = {"cos": cosine_similarity, "euc": euclidian_sim, "dummy": dummy_sim}
    def __init__(
            self,
            n_obs_states,
            n_actions,
            plan_steps,
            mt_lr,
            jn_lr,
            mt_beta,
            use_cluster_size_bias,
            use_memory_trace,
            update_period,
            gamma,
            reward_lr,
            inverse_temp,
            exploration_eps,
            trace_gamma,
            sim_metric,
            sleep_period,
            merge_iterations,
            split_iterations,
            clusters_per_obs,
            top_percent_to_merge,
            seed
    ):
        self.n_obs_states = n_obs_states
        self.n_actions = n_actions
        self.plan_steps = plan_steps
        self.update_period = update_period
        self.sleep_period = sleep_period
        self.merge_iterations = merge_iterations
        self.split_iterations = split_iterations
        # +1 for the initial state (the last state)
        self.first_order_transitions = np.zeros((n_actions, n_obs_states + 1, n_obs_states + 1))
        self.first_level_transitions = [dict() for _ in range(n_actions)]
        self.state_to_memory_trace = dict()
        self.state_to_sf = dict()
        self.second_level_transitions = [dict() for _ in range(n_actions)]
        self.cluster_to_states = dict()
        self.cluster_to_obs = dict()
        self.state_to_cluster = dict()
        self.cluster_to_entropy = dict()
        self.obs_to_clusters = {obs: set() for obs in range(self.n_obs_states)}
        self.mt_merge_thresholds = dict()
        self.joint_norm = SSValue(1.0, jn_lr, mean_ini=1.0)
        self.mt_lr = mt_lr
        self.mt_beta = mt_beta
        self.use_cluster_size_bias = use_cluster_size_bias
        self.use_memory_trace = use_memory_trace
        self.top_percent_to_merge = top_percent_to_merge

        self.state = (-1, -1)
        self.cluster = {(-1, -1): 1.0}
        self.cluster_to_states[-1] = {(-1, -1)}
        self.state_to_cluster[(-1, -1)] = -1
        self.cluster_to_obs[-1] = -1

        self.clusters_allocated = clusters_per_obs > 0
        if self.clusters_allocated:
            for obs_state in self.obs_to_clusters:
                clusters = set(
                        range(
                            obs_state * clusters_per_obs, (obs_state + 1) * clusters_per_obs
                        )
                    )
                self.obs_to_clusters[obs_state].update(
                    clusters
                )
                for c in clusters:
                    self.cluster_to_states[c] = set()
                    self.cluster_to_obs[c] = obs_state

        self.gamma = gamma
        self.trace_gamma = trace_gamma
        self.sim_func = self.similarities[sim_metric]
        self.reward_lr = reward_lr
        self.rewards = np.zeros(self.n_obs_states, dtype=np.float32)
        self.num_clones = np.zeros(self.n_obs_states, dtype=np.uint32)
        self.action_values = np.zeros(self.n_actions)

        if isinstance(self.trace_gamma, float):
            self.trace_gamma = np.array([self.trace_gamma])
        else:
            self.trace_gamma = np.array(self.trace_gamma)

        self.memory_trace = np.zeros((len(self.trace_gamma), self.n_obs_states))

        self.first_order_error = 0
        self.first_order_acc = 0
        self.first_level_acc = 0
        self.second_level_error = 0
        self.second_level_acc = 0
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
        self.first_order_error = 0
        self.first_order_acc = 0
        self.first_level_acc = 0
        self.second_level_error = 0
        self.second_level_acc = 0
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
        self.memory_trace = self.trace_gamma[:, None] * self.memory_trace

        obs_state = int(obs_state[0])
        predicted_state = self.first_level_transitions[action].get(self.state)
        self.first_level_none = float(predicted_state is None)
        if (predicted_state is None) or (predicted_state[0] != obs_state):
            self.first_level_acc = 0
            current_state = self._new_state(obs_state)
        else:
            self.first_level_acc = 1
            current_state = predicted_state

        # first order baseline
        obs_probs = self.first_order_transitions[action][self.state[0]]
        obs_probs = normalize(obs_probs).squeeze()
        self.first_order_error = - np.log(obs_probs[obs_state] + EPS)
        self.first_order_acc = np.argmax(obs_probs) == obs_state

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

        self.second_level_none = 1 - obs_probs.sum()
        self.second_level_error = - np.log(obs_probs[obs_state] + EPS)
        self.second_level_acc = np.argmax(obs_probs) == obs_state
        # posterior update of predicted clusters (IMPORTANT! we didn't do that for previous tests)
        predicted_clusters = {c: p for c, p in predicted_clusters.items() if c[0] == obs_state}
        norm = sum(list(predicted_clusters.values())) + EPS
        predicted_clusters = {c: p/norm for c, p in predicted_clusters.items()}

        self.generalised = self.second_level_acc > self.first_level_acc

        # state induced cluster
        cluster = self.state_to_cluster.get(current_state)
        if cluster is not None:
            predicted_clusters = {(current_state[0], cluster): 1.0}

        current_clusters = predicted_clusters

        if learn:
            current_mem_trace = self.memory_trace.flatten()
            if current_state is not None:
                self.state_to_memory_trace[current_state] = current_mem_trace

            if (self.state is not None) and (current_state is not None):
                self.first_level_transitions[action][self.state] = current_state
                self.first_order_transitions[action][self.state[0]][current_state[0]] += 1

            if cluster is None:
                # add state to a cluster with the most similar memory trace
                winner = self.assign_cluster(obs_state, current_mem_trace, predicted_clusters)

                # or create a new cluster
                if winner is None:
                    winner = self.cluster_counter
                    self.cluster_counter += 1
                    self.cluster_to_states[winner] = set()
                    self.cluster_to_obs[winner] = obs_state
                    self.obs_to_clusters[obs_state].add(winner)

                self.cluster_to_states[winner].add(current_state)
                self.state_to_cluster[current_state] = winner

            if (self.time_step % self.update_period) == 0:
                self._update_second_level()

            if (self.time_step % self.sleep_period) == 0:
                self.sleep_phase(self.merge_iterations, self.split_iterations)

        self.memory_trace += obs_dense[None]
        self.state = current_state
        # TODO cluster and current_cluster may be incongruent
        self.cluster = current_clusters
        self.time_step += 1

    def assign_cluster(self, obs_state, mem_trace, predicted_clusters):
        # transition-based predictions
        candidates = list(self.obs_to_clusters[obs_state])
        # TODO check case with one candidate
        if len(candidates) < 2:
            return None

        pred_probs = np.array(
            [predicted_clusters.get((obs_state, c), 0.0) for c in candidates]
        )
        candidates = np.array(candidates)

        if self.use_memory_trace:
            # memory-trace-based predictions
            traces = list()
            for c in candidates:
                trace = [self.state_to_memory_trace[s] for s in self.cluster_to_states[c]]
                trace = np.vstack(trace).mean(axis=0)
                traces.append(trace)

            means, stds = self.extract_thresholds(
                candidates, self.mt_merge_thresholds
            )
            means = np.array(means)
            stds = np.array(stds)

            traces = np.vstack(traces)
            scores = self.sim_func(traces, mem_trace[None])[0]
            self.update_thresholds(
                scores, candidates, self.mt_merge_thresholds, self.mt_lr
            )
            # standardise scores
            scores = (scores - means) / (stds + EPS)
            mt_probs = np.exp(self.mt_beta * scores)

            # filter noisy scores
            filter_scores = norm_cdf(scores)
            filter_mask = self._rng.random(len(filter_scores)) > filter_scores
            mt_probs[filter_mask] = 0.0
            mt_probs = mt_probs / (mt_probs.sum() + EPS)

            # main formula
            joint_probs = pred_probs * mt_probs
            joint_norm = joint_probs.sum()
            jm = joint_norm / self.joint_norm.mean
            q = (
                    np.power(joint_probs, jm) *
                    np.power(pred_probs + mt_probs, max(0, 1 - jm))
            )
            self.joint_norm.update(joint_norm)
        else:
            q = pred_probs

        q = normalize(q).squeeze()

        if self.use_cluster_size_bias:
            # cluster-size-based prior
            lengths = [
                len(self.cluster_to_states[c])
                for c in candidates
            ]
            lengths = np.array(lengths, dtype=np.float32)
            c_prior = lengths / (lengths.sum() + EPS)
            q *= c_prior

        q = normalize(q).squeeze()

        # decide if we create a new cluster
        new_cluster_prob = np.exp(entropy(q)) / len(candidates)
        if (self._rng.random() > new_cluster_prob) or self.clusters_allocated:
            winner = self._rng.choice(candidates, p=q)
        else:
            winner = None
        return winner

    @staticmethod
    def update_thresholds(score, clusters, thresholds, lr):
        for s, c in zip(score, clusters):
            if c in thresholds:
                thresholds[c].update(s)
            else:
                thresholds[c] = SSValue(1.0, lr, mean_ini=s)

    @staticmethod
    def extract_thresholds(clusters, thresholds):
        means = list()
        stds = list()
        for c in clusters:
            if c in thresholds:
                sv = thresholds[c]
                means.append(sv.mean)
                stds.append(sv.std)
            else:
                means.append(0.0)
                stds.append(1.0)
        return means, stds

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
            sf, steps, gf = self.generate_sf({predicted_state})
            self.goal_found = gf or self.goal_found
            planning_steps += steps
            self.action_values[action] = np.sum(sf * self.rewards)

        self.sf_steps = planning_steps / self.n_actions
        return self.action_values

    def generate_sf(self, initial_state: set, early_stop: bool = True, expand_clusters: bool = False):
        sf = np.zeros(self.n_obs_states, dtype=np.float32)
        goal_found = False

        if (initial_state is None) or (len(initial_state) == 0):
            return sf, 0, False

        predicted_states = initial_state
        sf, _ = self._states_obs_dist(initial_state)

        discount = self.gamma
        i = -1
        for i in range(self.plan_steps):
            # uniform strategy
            predicted_states = set().union(
                *[self._predict(predicted_states, a, expand_clusters) for a in range(self.n_actions)]
            )
            if len(predicted_states) == 0:
                break

            obs_probs, obs_states = self._states_obs_dist(predicted_states)
            sf += discount * obs_probs

            if np.any(
                self.rewards[list(obs_states)] > 0
            ):
                goal_found = True
                if early_stop:
                    break

            discount *= self.gamma

        return sf, i+1, goal_found

    def sleep_phase(self, merge_iterations, split_iterations):
        for _ in range(merge_iterations):
            n_cls = self.num_clusters
            n_clusters = [len(self.obs_to_clusters[obs]) for obs in range(self.n_obs_states)]
            n_clusters = np.array(n_clusters, dtype=np.float32)

            # sample obs state to start replay from
            probs = np.clip(n_clusters - 1, 0, None)
            norm = probs.sum()
            if norm == 0:
                # no pairs of clusters
                break
            probs /= norm
            obs_state = self._rng.choice(self.n_obs_states, p=probs)

            clusters = np.array(list(self.obs_to_clusters[obs_state]))
            sfs = list()
            for c in clusters:
                states = self.cluster_to_states[c]
                sf, _, _ = self.generate_sf(states, early_stop=False)
                sfs.append(sf)

            sfs = np.vstack(sfs)
            # merge candidates
            pairs = np.triu_indices(sfs.shape[0], k=1)
            scores = self.sim_func(sfs)
            scores = scores[pairs].flatten()
            pairs = np.column_stack(pairs)
            # merge most similar pairs
            k = int(self.top_percent_to_merge * len(scores))
            top_k_inds = np.argpartition(scores, -k)[-k:]
            pairs_to_merge = pairs[top_k_inds]

            for i in range(pairs_to_merge.shape[0]):
                pair = pairs[i]
                if pair[0] != pair[1]:
                    self._merge_clusters(clusters[pair[0]], clusters[pair[1]], obs_state)
                    # replace merged clusters ids
                    pairs[pairs == pair[1]] = pair[0]

            prefix = "sleep/merge/"
            try:
                wandb.log(
                    {
                        prefix + 'num_candidate_pairs': k,
                        prefix + 'delta_num_clusters': self.num_clusters - n_cls,
                        prefix + 'mean_sf_sim': scores.mean(),
                        prefix + 'mean_sf_std': scores.std()
                    }
                )
            except wandb.errors.Error:
                pass

        self._update_second_level()

        for _ in range(split_iterations):
            n_cls = self.num_clusters
            # sample clusters proportionally to
            # the entropy of their predictions
            cluster_entropies = np.array(list(self.cluster_to_entropy.values()), dtype=np.float32)
            clusters = np.array(list(self.cluster_to_entropy.keys()), dtype=np.int32)
            probs = 1 - np.exp(-cluster_entropies)
            g = self._rng.random(len(probs))
            clusters_to_split = clusters[g<probs]

            if len(clusters_to_split) == 0:
                break

            n_split = 0
            for cluster in clusters_to_split:
                if self._split_cluster(cluster):
                    n_split += 1

            if n_split:
                self._update_second_level()

            prefix = "sleep/split/"
            try:
                wandb.log(
                    {
                        prefix + 'num_candidates': len(clusters_to_split),
                        prefix + 'delta_num_clusters': self.num_clusters - n_cls,
                        prefix + 'num_split': n_split,
                        prefix + 'mean_entropy': cluster_entropies.mean()
                    }
                )
            except wandb.errors.Error:
                pass

    def _split_cluster(self, cluster_id):
        mask = self._test_cluster(cluster_id)
        states = np.array(list(self.cluster_to_states[cluster_id]))
        obs_state = self.cluster_to_obs[cluster_id]
        old_cluster = states[mask]
        new_cluster = states[~mask]
        if len(new_cluster) == 0:
            return
        # update old cluster
        self.cluster_to_states[cluster_id] = {tuple(s) for s in old_cluster}
        # create new cluster
        new_cluster_id = self.cluster_counter
        self.cluster_counter += 1
        self.cluster_to_states[new_cluster_id] = {tuple(s) for s in new_cluster}

        for s in self.cluster_to_states[new_cluster_id]:
            self.state_to_cluster[s] = new_cluster_id

        self.cluster_to_obs[new_cluster_id] = obs_state
        self.obs_to_clusters[obs_state].add(new_cluster_id)
        return new_cluster_id

    def _test_cluster(self, cluster_id):
        states = self.cluster_to_states[cluster_id]
        test = np.ones(len(states)).astype(np.bool8)

        for d_a in self.first_level_transitions:
            clusters = np.full(len(states), fill_value=np.nan)
            for pos, s in enumerate(states):
                ps = d_a.get(s)
                clusters[pos] = self.state_to_cluster.get(ps, np.nan)
            # detect contradiction
            empty = np.isnan(clusters)
            cls, counts = np.unique(clusters[~empty], return_counts=True)
            if len(counts) > 0:
                test = (clusters == cls[np.argmax(counts)]) | empty
        return test

    def _merge_clusters(self, c1, c2, obs_state):
        new_states = self.cluster_to_states[c1].union(self.cluster_to_states[c2])
        self.cluster_to_states[c1] = new_states

        states = self.cluster_to_states.pop(c2)
        for s in states:
            self.state_to_cluster[s] = c1

        self.obs_to_clusters[obs_state].remove(c2)

        if c2 in self.cluster_to_entropy:
            self.cluster_to_entropy.pop(c2)

        for d_a in self.second_level_transitions:
            if c2 in d_a:
                d_a.pop(c2)

    def _update_second_level(self):
        for d_a in self.second_level_transitions:
            d_a.clear()
        for cluster_id in self.cluster_to_states:
            # update transition matrix
            cluster_entropy = 0
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
                    cluster_entropy += entropy(probs)
                    cluster = (self.cluster_to_obs[cluster_id], cluster_id)
                    pred_clusters = {(self.cluster_to_obs[pc], pc, p) for pc, p in zip(predicted_clusters, probs)}
                    d_a[cluster] = pred_clusters

            self.cluster_to_entropy[cluster_id] = cluster_entropy

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

    def _states_obs_dist(self, states: set):
        obs_states = np.array(
            self._convert_to_obs_states(states),
            dtype=np.uint32
        )
        obs_states, counts = np.unique(obs_states, return_counts=True)
        obs_probs = np.zeros(self.n_obs_states, dtype=np.float32)
        obs_probs[obs_states] = counts
        obs_probs /= (obs_probs.sum() + EPS)
        return obs_probs, obs_states

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
    def draw_transition_graph(self, threshold=0.2):
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