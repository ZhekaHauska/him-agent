#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.experiments.cognitive_maps.trace import TraceBasedLoc
from hima.envs.gridworld import GridWorld
from hima.common.config.base import read_config
import numpy as np
import wandb
import os


def main():
    config = read_config('configs/runner.yaml')
    if config['seed'] is None:
        config['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    trace, norm_trace, state = get_data(config)

    np.savez('data/test.npz', trace=trace, norm_trace=norm_trace, state=state)
    # clusterize


def get_data(config, overrides=None):
    rng = np.random.default_rng(config['seed'])

    config['env'] = read_config(config['env'])
    config['model'] = read_config(config['model'])

    # override settings
    if overrides is not None:
        for key, value in overrides.items():
            keys = key.split('.')
            cfg = config
            for x in keys[:-1]:
                cfg = cfg[x]
            cfg[keys[-1]] = value

    env = GridWorld(**config['env'], seed=config['seed'])
    model = TraceBasedLoc(n_obs_states=env.n_colors, **config['model'], seed=config['seed'])
    model.learn = False

    trace = []
    norm_trace = []
    state = []

    for epoch in range(config['n_epochs']):
        env.reset()
        model.reset()

        for step in range(config['n_steps']):
            observation, _, _ = env.obs()
            observation = observation.flatten()[0]
            obs_state = np.flatnonzero(env.unique_colors == observation)
            model.observe(obs_state)

            trace.append(model.observation_trace.copy())
            norm_trace.append(model.norm_observation_trace.copy())
            state.append((env.r, env.c))

            env.act(rng.integers(0, len(env.actions)))
            env.step()

    return trace, norm_trace, state


def clusterize():
    ...


if __name__ == '__main__':
    main()
