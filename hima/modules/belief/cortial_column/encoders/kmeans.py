#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import pickle

import numpy as np
from sklearn.cluster import KMeans
from hima.modules.belief.cortial_column.encoders.base import BaseEncoder


class KMeansEncoder(BaseEncoder):
    model: KMeans

    def __init__(self, path):
        with open(path, 'rb') as file:
            self.model = pickle.load(file)
        self.n_vars = 1
        self.n_states = self.model.n_clusters

    def encode(self, input_: np.ndarray, learn: bool) -> np.ndarray:
        return self.model.predict(input_.flatten()[None])
