#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.experiments.cognitive_maps.trace import TraceBasedLoc
from hima.envs.gridworld import GridWorld
from hima.common.config.base import read_config
import numpy as np
import sys


def main():
    seed = 123
    n_steps = 50
    n_epochs = 100

    env_conf = read_config('configs/gridworld.yaml')
    model_conf = read_config('configs/trace.yaml')

    env = GridWorld(**env_conf, seed=seed)
    model = TraceBasedLoc(n_obs_states=env.n_colors, **model_conf, seed=seed)

    for epoch in range(n_epochs):
        env.reset()
        model.reset()

        for step in range(n_steps):
            observation, _, _ = env.obs()
            observation = observation.flatten()[0]
            obs_state = np.flatnonzero(env.unique_colors == observation)
            model.observe(obs_state)


if __name__ == '__main__':
    main()
