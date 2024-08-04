#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.experiments.cognitive_maps.toy_dhtm import ToyDHTM
from hima.envs.gridworld import GridWorld
from hima.common.config.base import read_config
import matplotlib.pyplot as plt
import seaborn as sns
import colormap

import numpy as np


if __name__ == '__main__':
    config = read_config('configs/dhtm_runner.yaml')
    if config['seed'] is None:
        config['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    rng = np.random.default_rng(config['seed'])

    config['env'] = read_config(config['env'])
    config['dhtm'] = read_config(config['dhtm'])

    env = GridWorld(**config['env'], seed=config['seed'])

    config['dhtm']['n_obs_states'] = env.n_colors
    config['dhtm']['n_actions'] = len(env.actions)
    dhtm = ToyDHTM(**config['dhtm'])

    label_counts = np.zeros((dhtm.n_hidden_states, env.h * env.w))
    states_visited = list()

    for epoch in range(config['n_epochs']):
        env.reset()

        if len(states_visited) > 0:
            for hidden_state, state in zip(dhtm.state_buffer, states_visited):
                label_counts[hidden_state, state] += 1

        dhtm.reset(env.get_true_map())
        states_visited.clear()

        for step in range(config['n_steps']):
            observation, _, _ = env.obs()
            states_visited.append(env.c + env.r * env.w)
            observation = observation.flatten()[0]
            obs_state = np.flatnonzero(env.unique_colors == observation)[0]
            action = rng.integers(0, len(env.actions))
            dhtm.observe(obs_state, action, (env.r, env.c))

            env.act(action)
            env.step()

    labels = np.argmax(label_counts, axis=-1)
    labels = [(s//env.w, s % env.w) for s in labels]
    dhtm.draw_graph('graph.png', connection_threshold=0.6, activation_threshold=5, labels=labels)
    m = env.get_true_map().astype(np.float32)
    m[m < 0] = np.nan
    sns.heatmap(m, cmap=colormap.cmap_builder('Pastel1'), annot=True, cbar=False, vmin=0, vmax=dhtm.n_obs_states)
    plt.savefig('map.png')
