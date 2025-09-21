import numpy as np
import pickle as pkl
import random as rd
import torch
from sklearn.cluster import KMeans, DBSCAN, Birch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def cluster_latent_vectors(latent_vectors, 
                           n_clusters, 
                           cluster_model='kmeans',
                        ):

    if latent_vectors.ndim == 3:
        latent_vectors = latent_vectors.reshape(-1, latent_vectors.shape[2])
    match cluster_model:
        case 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, algorithm='elkan')
        case 'dbscan':
            model = DBSCAN()
        case 'birch':
            model = Birch(n_clusters=n_clusters)
        case _:
            raise ValueError(f'{cluster_model} is not from knowm models list (kmeans, dbscan, birch)')
    cluster_labels = model.fit_predict(latent_vectors)
    
    return cluster_labels, model

def build_transition_matrix(states, n_clusters, actions=None, mapping=None, model=None, cluster_labels=None):

    cluster_labels = cluster_labels.reshape(states.shape[0], -1)
    if actions is not None:
        transition_matrix = np.zeros((actions.shape[2], n_clusters, n_clusters))
    else:
        transition_matrix = np.zeros((n_clusters, n_clusters))
    
    if cluster_labels is None and model is None:
        raise ValueError('model and cluster_labels must not be None at the same time')
    
    for j, sequence in enumerate(states):
        if cluster_labels is not None:
            labels = cluster_labels[j]
        else:
            labels = model.predict(sequence)
        for i in range(len(sequence) - 1):
            if mapping is not None:
                current_cluster = mapping[labels[i]]
                next_cluster = mapping[labels[i + 1]]
            else:
                current_cluster = labels[i]
                next_cluster = labels[i + 1]
            if actions is not None:
                action = actions[j, i]
                action_idx = np.argmax(action)
                transition_matrix[action_idx, current_cluster, next_cluster] += 1
            else:
                transition_matrix[current_cluster, next_cluster] += 1
    
    row_sums = transition_matrix.sum(axis=2, keepdims=True)
    row_sums[row_sums == 0] = 1000
    transition_matrix /= row_sums
    
    return transition_matrix

def generate_true_transition_matrix(field):
    n = len(field)
    total_states = n * n
    
    transition_matrix = np.zeros((4, total_states, total_states), dtype=float)

    direction_vectors = [
        (-1, 0),
        (0, 1),
        (1, 0), 
        (0, -1) 
    ]
    
    for i in range(n):
        for j in range(n):
            current_state = i * n + j
            
            if field[i][j] < 0:
                continue

            for action in range(4):
                dx, dy = direction_vectors[action]
                ni, nj = i + dx, j + dy

                if 0 <= ni < n and 0 <= nj < n:

                    if field[ni][nj] >= 0:

                        target_state = ni * n + nj
                        transition_matrix[action, current_state, target_state] = 1.0
                    else:
                        transition_matrix[action, current_state, current_state] = 1.0
                else:
                    transition_matrix[action, current_state, current_state] = 1.0
    
    sums = transition_matrix.sum(axis=2, keepdims=True)
    sums[sums == 0] = 1.0
    transition_matrix /= sums
    return transition_matrix

def plot_transition_matrices(predicted_matrix, true_matrix=None, titles=None, figsize=(15, 6)):

    if titles is None:
        titles = ['Predicted Transition Matrix', 'True Transition Matrix']

    if true_matrix is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plot_single_matrix(ax, predicted_matrix, titles[0])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        plot_single_matrix(ax1, predicted_matrix, titles[0])
        plot_single_matrix(ax2, true_matrix, titles[1])

        fig.suptitle('Transition Matrices Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_single_matrix(ax, matrix, title):

    im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)

    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.grid(False)

def align_transition_matrices(n_states, true_labels, cluster_labels):

    confusion_matrix = np.zeros((n_states, n_states))
    
    for true_label, cluster_label in zip(true_labels, cluster_labels):
        confusion_matrix[true_label, cluster_label] += 1
    
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    return dict(zip(col_ind, row_ind))

def make_obs_to_state_matrix(transition_matrix, room):
    out_matrix = np.zeros((transition_matrix.shape[0], len(np.unique(room))))
    room = room.flatten()
    for state in range(len(room)):
        out_matrix[:, room[state]] += transition_matrix[:, state]

    sums = out_matrix.sum(axis=-1, keepdims=True)
    sums[sums == 0] = 1000
    out_matrix /= sums
    return out_matrix
