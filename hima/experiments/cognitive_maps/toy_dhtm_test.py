#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.experiments.cognitive_maps.toy_dhtm import ToyDHTM
from hima.envs.gridworld import GridWorld
from hima.common.config.base import read_config

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

    for epoch in range(config['n_epochs']):
        env.reset()
        dhtm.reset()

        for step in range(config['n_steps']):
            observation, _, _ = env.obs()
            observation = observation.flatten()[0]
            obs_state = np.flatnonzero(env.unique_colors == observation)
            action = rng.integers(0, len(env.actions))
            dhtm.observe(obs_state, action)

            env.act(action)
            env.step()
