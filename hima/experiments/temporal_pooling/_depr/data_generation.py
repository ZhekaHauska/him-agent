#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from tempfile import NamedTemporaryFile
from typing import Optional, Iterator

import numpy as np
import ruamel.yaml as yaml
from numpy.random import Generator

from animal_ai_v1_pickle import through_v1, collect_data, list_to_np


class SequenceSelector:
    n_elements: int
    regime: str
    seed: Optional[int]

    indices: np.ndarray

    def __init__(self, n_elements: int, regime: str, seed: int = None):
        self.n_elements = n_elements
        self.regime = regime
        self.seed = seed
        self.indices = self._select_order()

    def reshuffle(self):
        self.seed = np.random.default_rng(self.seed).integers(100000)
        self.indices = self._select_order()

    def __iter__(self) -> Iterator[int]:
        if self.regime == 'unordered':
            # re-shuffle each time making the order to be random
            self.reshuffle()

        return iter(self.indices)

    def _select_order(self) -> np.ndarray:
        if self.regime == 'ordered':
            return np.arange(self.n_elements)
        elif self.regime == 'shuffled' or self.regime == 'unordered':
            rng = np.random.default_rng(self.seed)
            return rng.permutation(self.n_elements)
        else:
            raise KeyError(f'{self.regime} is not supported')


def generate_data(n, n_actions, n_states, randomness=1.0, seed=0):
    raw_data = list()
    np.random.seed(seed)
    seed_seq = np.random.randint(0, n_actions, n_states)
    raw_data.append(seed_seq.copy())
    n_replace = int(n_states * randomness)
    for i in range(1, n):
        new_seq = np.random.randint(0, n_actions, n_states)
        if randomness == 1.0:
            raw_data.append(new_seq)
        else:
            indices = np.random.randint(0, n_states, n_replace)
            seed_seq[indices] = new_seq[indices]
            raw_data.append(seed_seq.copy())
    data = [list(zip(range(n_states), x)) for x in raw_data]
    return raw_data, data


def change_pos(room, pos):
    for _, arenas in room.items():
        for _, objects in arenas.items():
            for name, items in objects.items():
                if name == 'items':
                    for item in items:
                        if item['name'] == 'Agent':
                            for pref_name, pref in item.items():
                                if pref_name == 'positions':
                                    pref[0]['x'], pref[0]['z'] = int(pos[0]), int(pos[1])
    return room


def generate_random_positions_observations(room_conf: str, n_positions=5, seed=42):
    rng = np.random.default_rng(seed=seed)
    yml = yaml.YAML()

    positions_obs = []

    with open(room_conf) as fp:
        room = yml.load(fp)
        positions = rng.integers(2, 38, (n_positions, 2))

        for pos in positions:
            #####
            room = change_pos(room, pos)
            tmp_file = NamedTemporaryFile()
            yml.dump(room, tmp_file)
            data = collect_data(tmp_file.name)
            positions_obs.append(data)
            #####
        return positions_obs


def generate_random_positions_observations_v1_output(room_conf: str, n_positions=5, seed=42):
    with open('../configs/v1.yaml', 'r') as yml_v1:
        v1_conf = yaml.safe_load(yml_v1)['v1_config']

    v1_outputs = []
    positions_obs = generate_random_positions_observations(room_conf, n_positions, seed)

    for obs in positions_obs:
        v1_outputs.append(through_v1(list_to_np(obs), v1_conf))

    return v1_outputs
