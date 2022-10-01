#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import pickle

import matplotlib.pyplot as plt
import numpy as np
import wandb

from data_generation import generate_random_positions_observations_v1_output
from hima.experiments.temporal_pooling.data_generation import generate_data


def _run_tests():
    wandb.login()
    np.random.seed(31)

    n_actions = 4
    n_states = 25

    row_data, data = generate_data(5, n_actions, n_states, randomness=0.7)
    np.random.shuffle(data)

    plt.figure(figsize=(15, 8))
    plt.gca().set_facecolor('lightgrey')

    v1_out = generate_random_positions_observations_v1_output(
        '/home/ivan/htm/him-agent/hima/experiments/temporal_pooling/configs/aai_rooms/different_points/l.yml',
        3
    )

    with open('rand_pos.pkl', 'wb') as f:
        pickle.dump(v1_out, f)


if __name__ == '__main__':
    _run_tests()
