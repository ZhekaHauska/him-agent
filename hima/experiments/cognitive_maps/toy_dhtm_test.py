#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import wandb

from hima.experiments.cognitive_maps.toy_dhtm import ToyDHTM
from hima.envs.gridworld import GridWorld
from hima.common.config.base import read_config
import matplotlib.pyplot as plt
import seaborn as sns
import colormap

import numpy as np


def get_true_transitions(n_clones, env: GridWorld):
    n_states = len(env.unique_colors)*n_clones
    transition_matrix = np.zeros((len(env.actions), n_states, n_states))
    true_map = env.get_true_map()
    obs_counts = np.zeros_like(env.unique_colors)
    clones = np.zeros_like(true_map)
    for r in range(env.h):
        for c in range(env.w):
            obs = true_map[r, c]
            clones[r, c] = obs_counts[obs]
            obs_counts[obs] += 1
    for r in range(env.h):
        for c in range(env.w):
            obs = true_map[r, c]
            clone = clones[r, c]
            u_state = clone + obs * n_clones
            for a in env.actions:
                env.r, env.c = r, c
                env.act(a)
                env.step()
                obs = true_map[env.r, env.c]
                clone = clones[env.r, env.c]
                v_state = clone + obs * n_clones
                transition_matrix[a, u_state, v_state] = 1
    return transition_matrix


if __name__ == '__main__':
    config = read_config('configs/dhtm_runner.yaml')
    if config['seed'] is None:
        config['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    rng = np.random.default_rng(config['seed'])

    config['env'] = read_config(config['env_path'])
    config['dhtm'] = read_config(config['dhtm_path'])

    log = config.pop('log')
    project_name = config.pop('project_name')
    if log:
        logger = wandb.init(
            project=project_name,
            config=config
        )
    else:
        logger = None

    env = GridWorld(**config['env'], seed=config['seed'])

    config['dhtm']['n_obs_states'] = env.n_colors
    config['dhtm']['n_actions'] = len(env.actions)
    dhtm = ToyDHTM(**config['dhtm'])

    if config['set_true_transitions']:
        true_transition_matrix = get_true_transitions(dhtm.n_clones, env).astype(
            dhtm.transition_counts.dtype
        )
        dhtm.transition_counts = true_transition_matrix
        dhtm.activation_counts = dhtm.transition_counts.sum(axis=0).sum(axis=-1).flatten(
        ).astype(dhtm.activation_counts.dtype)

    if 'initial_pos' in config:
        init_r, init_c = config['initial_pos']
    else:
        init_r, init_c = None, None

    label_counts = np.zeros((dhtm.n_hidden_states, env.h * env.w))
    states_visited = list()
    strategy = config.get('strategy', None)
    action_step = 0
    total_steps = 0

    for epoch in range(config['n_epochs']):
        env.reset(init_r, init_c)

        if len(states_visited) > 0:
            for hidden_state, state in zip(dhtm.state_buffer, states_visited):
                label_counts[hidden_state, state] += 1

        replay_surprise = dhtm.replay()
        if log:
            logger.log(
                {
                    'replay_surprise': replay_surprise
                },
                step=total_steps
            )
        dhtm.reset(env.get_true_map())
        states_visited.clear()

        for step in range(config['n_steps']):
            observation, _, _ = env.obs()
            states_visited.append(env.c + env.r * env.w)
            observation = observation.flatten()[0]
            obs_state = np.flatnonzero(env.unique_colors == observation)[0]

            if strategy is not None:
                action = strategy[action_step]
                action_step += 1
                action_step %= len(strategy)
            else:
                action = rng.integers(0, len(env.actions))

            surprise = - np.log(np.clip(dhtm.predict()[obs_state], 1e-24, 1.0))
            dhtm.observe(obs_state, action, (env.r, env.c))

            env.act(action)
            env.step()

            if log:
                logger.log({
                    'clones_used': np.count_nonzero(dhtm.activation_counts > 0),
                    'surprise': surprise
                }, step=total_steps)

            total_steps += 1

    labels = np.argmax(label_counts, axis=-1)
    labels = [(s//env.w, s % env.w) for s in labels]

    graph_params = config.get('graph_params', None)
    if graph_params is not None:
        dhtm.draw_graph(**graph_params, labels=labels)

    map_path = config.get('map_path', None)
    if map_path is not None:
        m = env.get_true_map().astype(np.float32)
        m[m < 0] = np.nan
        sns.heatmap(
            m,
            cmap=colormap.cmap_builder('Pastel1'),
            annot=True,
            cbar=False,
            vmin=0,
            vmax=dhtm.n_obs_states
        )
        plt.savefig('map.png')
