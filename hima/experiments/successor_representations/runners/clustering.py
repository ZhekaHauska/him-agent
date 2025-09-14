#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle

from sklearn.cluster import HDBSCAN, OPTICS, AgglomerativeClustering

from hima.agents.episodic_control.agent import ECAgent
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def dummy_sim(X, Y=None):
    if Y is None:
        return np.zeros((X.shape[0], X.shape[0]))
    else:
        return np.zeros((X.shape[0], Y.shape[0]))


def prepare_data(agent: ECAgent, embedding_type='sf', plan_steps=None, gamma=None):
    if gamma:
        agent.gamma = gamma
    if plan_steps:
        agent.plan_steps = plan_steps

    embds = list()
    for s in tqdm(agent.state_labels):
        embd = list()
        if "mt" in embedding_type:
            trace = agent.state_to_memory_trace[s]
            embd.append(trace)
        if "sf" in embedding_type:
            sf, _, _ = agent.generate_sf({s}, early_stop=False)
            embd.append(sf)
        embds.append(np.concatenate(embd))
    return np.vstack(embds), np.array(list(agent.state_labels.values())), np.array([s[0] for s in agent.state_labels])


def cluster_purity(labels, true_labels):
    assert len(labels) == len(true_labels)

    sorted_ids = np.argsort(labels)
    labels = labels[sorted_ids]
    true_labels = true_labels[sorted_ids]

    labels, split_ids = np.unique(labels, return_index=True)
    result = []
    for cluster in np.array_split(true_labels, split_ids):
        tl, counts = np.unique(cluster, return_counts=True)
        if len(cluster) <= 1:
            continue
        else:
            score = np.max(counts) / counts.sum()
            result.append(score)
    return np.array(result)

def load_agent(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

methods = [
    (HDBSCAN, dict(min_cluster_size=500, metric='correlation')),
    (HDBSCAN, dict(min_cluster_size=100, metric='correlation')),
    (HDBSCAN, dict(min_cluster_size=10, metric='correlation')),
    # (OPTICS, dict(metric='correlation')),
    #(AgglomerativeClustering, dict(n_clusters=10, distance_threshold=None, metric='cosine', linkage='average')),
    # (AgglomerativeClustering, dict(n_clusters=None, distance_threshold=0.3, metric='euclidean')),
    # (AgglomerativeClustering, dict(n_clusters=None, distance_threshold=0.1, metric='cosine', linkage='average'))
]

def main():
    agent: ECAgent = load_agent('../logs/agent_wandering-wave-141.pkl')
    # cache_path = None
    cache_path = '../logs/data_cache_sf_0.8_50.npz'

    if cache_path:
        print('use cached data')
        cache = np.load(cache_path)
        X, true_labels, obs = cache['X'], cache['true_labels'], cache['obs']
    else:
        print('prepare data:')
        cfg = dict( embedding_type='sf', gamma=0.8, plan_steps=50)
        X, true_labels, obs = prepare_data(agent, **cfg)
        np.savez('../logs/data_cache_{}_{}_{}.npz'.format(*cfg.values()),
            X=X, true_labels=true_labels, obs=obs
        )

    obs_states, obs_counts = np.unique(obs, return_counts=True)
    fig, axs = plt.subplots(nrows=3, ncols=len(obs_states))
    fig.tight_layout()
    ax = axs[0][0]
    ax.set_title('obs counts')
    sns.histplot(obs, ax=ax)

    for method in methods:
        cls, cfg = method
        m = cls(**cfg)
        print(f'fit data using {m} ...')
        for obs_state in obs_states:
            obs_mask = obs==obs_state
            labels = m.fit(X[obs_mask]).labels_
            ax = axs[1][obs_state]
            ax.set_title(f'cluster sizes obs={obs_state}')
            sns.histplot(labels[labels > -1], ax=ax, alpha=0.25, label=str(method))
            purity = cluster_purity(labels, true_labels[obs_mask])
            ax = axs[2][obs_state]
            ax.set_title(f'purity obs={obs_state}')
            sns.histplot(purity, ax=ax, alpha=0.25, label=str(method))
    plt.legend(loc='upper right', draggable=True)
    plt.show()

if __name__ == '__main__':
    main()
