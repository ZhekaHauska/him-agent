#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from unittest import TestCase
from hima.modules.htm.dchmm import DCHMM
import yaml
import numpy as np


class TestDCHMM(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDCHMM, self).__init__(*args, **kwargs)

        with open('configs/dchmm_default.yaml', 'r') as file:
            self.config = yaml.load(file, Loader=yaml.Loader)

        self.dchmm = DCHMM(
            **self.config
        )

    def test_cat_sampler(self):
        probs = self.dchmm._rng.dirichlet(
                    alpha=[1.0]*self.dchmm.n_hidden_states,
                    size=self.dchmm.n_hidden_vars
                )

        sample = self.dchmm._sample_categorical_variables(probs)

        self.assertTupleEqual(sample.shape, (self.dchmm.n_hidden_vars,))
        self.assert_(np.all((sample < self.dchmm.n_hidden_states) & (sample >= 0)))

    def test_cell_sampler(self):
        self.dchmm.prediction = self.dchmm._rng.dirichlet(
                    alpha=[1.0]*self.dchmm.n_hidden_states,
                    size=self.dchmm.n_hidden_vars
                ).flatten()

        obs = np.arange(self.dchmm.n_obs_vars) * self.dchmm.n_obs_states + self.dchmm._rng.integers(
            low=0,
            high=self.dchmm.n_obs_states,
            size=self.dchmm.n_obs_vars
        )

        cells_in_columns = self.dchmm._get_cells_in_columns(obs)
        obs_factor = np.zeros_like(self.dchmm.forward_messages)
        obs_factor[cells_in_columns] = 1

        new_forward_messages = self.dchmm.prediction * obs_factor

        next_active_cells = self.dchmm._sample_cells(
            new_forward_messages,
            cells_in_columns
        )

        self.assert_(np.all(next_active_cells < self.dchmm.total_cells))

    def test_prediction(self):
        self.dchmm.predict()
