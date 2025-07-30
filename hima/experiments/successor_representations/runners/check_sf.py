#  Copyright (c) 2025 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle
import numpy as np
from collections import defaultdict

import pandas as pd
from hima.common.sdr import sparse_to_dense
from scipy.special import kl_div
from scipy.spatial.distance import euclidean, correlation, cosine

from hima.modules.belief.utils import normalize

EPS = 1e-24

def dummy_sim(x, y):
    return 0

def dkl_sim(x, y):
    x = normalize(x)
    y = normalize(y)
    return np.exp(-kl_div(x, y).sum())

def euc_sim(x, y):
    return np.exp(-euclidean(x, y))

def cos_sim(x, y):
    return 1 - cosine(x, y)

def corr_sim(x, y):
    return 1 - correlation(x, y)

SIM_FUNCS = {
    'dummy': dummy_sim,
    'dkl': dkl_sim,
    'euc': euc_sim,
    'cos': cos_sim,
    'corr': corr_sim,
}

def _convert_to_obs_states(states):
    return [s[0] for s in states if s is not None]

def _states_obs_dist(n_obs_states: int, states: set):
    obs_states = np.array(
        _convert_to_obs_states(states),
        dtype=np.uint32
    )
    obs_states, counts = np.unique(obs_states, return_counts=True)
    obs_probs = np.zeros(n_obs_states, dtype=np.float32)
    obs_probs[obs_states] = counts
    obs_probs /= (obs_probs.sum() + EPS)
    return obs_probs, obs_states

def _predict_first_level(state: set, action: int, transitions) -> set:
    state_expanded = state
    d_a = transitions[action]
    predicted_state = set()
    for s in state_expanded:
        if s in d_a:
            predicted_state.add(d_a[s])
    return predicted_state

def generate_sf(
        n_obs_states: int,
        n_actions: int,
        initial_state: set,
        steps: int,
        gamma: float,
        transitions,
):
    sf = np.zeros(n_obs_states, dtype=np.float32)

    if (initial_state is None) or (len(initial_state) == 0):
        return sf, 0, False

    predicted_states = initial_state
    sf, _ = _states_obs_dist(n_obs_states, initial_state)

    discount = gamma
    i = -1
    for i in range(steps):
        # uniform strategy
        predicted_states = set().union(
            *[
                _predict_first_level(predicted_states, a, transitions)
                for a in range(n_actions)
            ]
        )
        if len(predicted_states) == 0:
            break

        obs_probs, obs_states = _states_obs_dist(n_obs_states, predicted_states)
        sf += discount * obs_probs

        discount *= gamma

    return sf, i + 1


def get_perfect_sf(
    true_transition_matrix,
    true_emission_matrix,
    init_label,
    plan_steps,
    gamma,
    rng,
    noise=0,
):
    # compare to perfect sf
    T = np.mean(true_transition_matrix, axis=0)
    E = true_emission_matrix

    current_state = sparse_to_dense([init_label], size=T.shape[0])
    perfect_sf = E[init_label].copy()

    discount = gamma
    for i in range(plan_steps):
        current_state = current_state @ T
        obs_dist = current_state @ E
        if noise > 0:
            obs_dist = rng.dirichlet((1 / noise**2) * (obs_dist + EPS))
        perfect_sf += discount * obs_dist
        discount *= gamma
    return perfect_sf


def form_clusters(
        n_clusters_per_label,
        cluster_size,
        purity,
        all_states,
        labels,
        label_to_obs,
        obs_to_labels,
        rng
):
    # sort states into labels
    inds = np.arange(len(all_states))
    rng.shuffle(inds)
    label_to_states = {label: list() for label in label_to_obs}
    for i in inds:
        label_to_states[labels[all_states[i]]].append(all_states[i])

    # output
    label_to_clusters = dict()
    cluster_to_states = dict()

    # create clusters
    cluster_id = 0
    for label in label_to_obs:
        label_to_clusters[label] = set()
        obs = label_to_obs[label]
        obs_labels = set(obs_to_labels[obs])
        obs_labels.discard(label)
        for _ in range(n_clusters_per_label):
            cluster = set()
            pure_states = round(purity * cluster_size)

            for _ in range(pure_states):
                cluster.add(label_to_states[label].pop())
            for _ in range(cluster_size - pure_states):
                random_label = rng.choice(list(obs_labels))
                cluster.add(label_to_states[random_label].pop())

            label_to_clusters[label].add(cluster_id)
            cluster_to_states[cluster_id] = cluster
            cluster_id += 1

    return cluster_to_states, label_to_clusters

def eval_clusters(
        sim_func,
        n_obs_states,
        n_actions,
        steps,
        gamma,
        transitions,
        cluster_to_states,
        obs_to_labels,
        label_to_clusters,
        true_transition,
        true_emission,
        noise,
        rng
):
    # output
    accuracy = defaultdict()
    similarities = defaultdict()
    perfect_sf_sim = defaultdict()
    generated_steps = defaultdict()

    for obs in obs_to_labels:
        labels = list(obs_to_labels[obs])
        probe_sfs = list()
        candidate_sfs = list()
        for label in labels:
            probe, candidate = list(label_to_clusters[label])
            probe_sf, gen_steps = generate_sf(
                n_obs_states,
                n_actions,
                cluster_to_states[probe],
                steps,
                gamma,
                transitions
            )
            generated_steps[label] = gen_steps
            probe_sfs.append(probe_sf)
            candidate_sf, _ = generate_sf(
                n_obs_states,
                n_actions,
                cluster_to_states[candidate],
                steps,
                gamma,
                transitions
            )
            candidate_sfs.append(candidate_sf)

            perfect_sf = get_perfect_sf(
                true_transition,
                true_emission,
                label,
                steps,
                gamma,
                rng,
                noise
            )
            perfect_sf_sim[label] = sim_func(probe_sf, perfect_sf)

        sim = np.zeros((len(probe_sfs), len(candidate_sfs)))
        for i, probe_sf in enumerate(probe_sfs):
            for j, candidate_sf in enumerate(candidate_sfs):
                sim[i, j] = sim_func(probe_sf, candidate_sf)
            predicted_label = labels[np.argmax(sim[i])]
            true_label = labels[i]
            accuracy[true_label] = int(true_label == predicted_label)
        similarities[obs] = similarities
    return accuracy, similarities, perfect_sf_sim, generated_steps

def eval_params(
        cluster_size, purity, sim_func_name, steps, gamma, seeds,
        all_states, state_to_label, label_to_obs, obs_to_labels,
        n_obs_states, n_actions, transitions,
        true_transition, true_emission, noise
):
    # test different cluster splits
    results = list()
    for seed in seeds:
        rng = np.random.default_rng(seed)

        cluster_to_states, label_to_clusters = form_clusters(
            2, cluster_size, purity,
            all_states,
            state_to_label,
            label_to_obs,
            obs_to_labels,
            rng
        )

        accuracy, similarities, perfect_sf_sim, generated_steps = eval_clusters(
            SIM_FUNCS[sim_func_name],
            n_obs_states,
            n_actions,
            steps,
            gamma,
            transitions,
            cluster_to_states,
            obs_to_labels,
            label_to_clusters,
            true_transition,
            true_emission,
            noise,
            rng
        )

        results.append(pd.DataFrame(
                {
                    'acc': list(accuracy.values()),
                    'sim': list(perfect_sf_sim.values()),
                    'gen_steps': list(generated_steps.values()),
                    'label': list(accuracy.keys()),
                    'cluster_size': cluster_size,
                    'purity': purity,
                    'sim_func': sim_func_name,
                    'steps': steps,
                    'gamma': gamma,
                    'noise': noise,
                    'seed': seed
                }
            )
        )

    return pd.concat(results, ignore_index=True)

def prepare_data(path):
    with open(path, 'rb') as file:
        experience = pickle.load(file)

    (
        transitions,
        state_to_label,
        label_to_obs,
        true_transition,
        true_emission
    ) = (
        experience['first_level'],
        experience['state_labels'],
        experience['label_to_obs'],
        experience['true_transition'],
        experience['true_emission']
    )
    all_states = list(set().union(*[set(d_a.keys()) for d_a in transitions]))
    all_states.remove((-1, -1))

    # convert labels to obs
    obs_to_labels = defaultdict(set)
    for label, obs in label_to_obs.items():
        obs_to_labels[obs].add(label)

    n_obs_states = len(obs_to_labels)
    n_actions = len(transitions)
    return (
        n_obs_states,
        n_actions,
        transitions,
        state_to_label,
        label_to_obs,
        obs_to_labels,
        true_transition,
        true_emission,
        all_states
    )

def main(path, seeds):
    (
        n_obs_states,
        n_actions,
        transitions,
        state_to_label,
        label_to_obs,
        obs_to_labels,
        true_transition,
        true_emission,
        all_states
    ) = prepare_data(path)

    results = list()

    # test different parameters
    cluster_size = 100
    purity = 0.9
    gamma = 0.8
    steps = 25
    noise = 0
    sim_func_name = 'corr'

    for sim_func_name in SIM_FUNCS:
        result = eval_params(
            cluster_size, purity, sim_func_name, steps, gamma, seeds,
            all_states, state_to_label, label_to_obs, obs_to_labels,
            n_obs_states, n_actions, transitions,
            true_transition, true_emission, noise
        )

        results.append(result)
    data = pd.concat(results, ignore_index=True)

    print(data.head(10))


if __name__ == '__main__':
    main(
        '../logs/1k_ep_5x5_fixed_pos_agent_experience.pkl',
        [123, 223]
        # [123, 322, 57487, 3329, 9993]
    )