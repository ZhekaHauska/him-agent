#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from unittest import TestCase
from hima.modules.belief.dchmm import DCHMM
import yaml
import numpy as np


class TestDCHMM(TestCase):
    def setUp(self) -> None:
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

        cells_in_columns = self.dchmm._get_cells_for_observation(obs)
        obs_factor = np.zeros_like(self.dchmm.forward_messages)
        obs_factor[cells_in_columns] = 1

        self.dchmm.next_forward_messages = self.dchmm.prediction * obs_factor

        next_active_cells = self.dchmm._sample_cells(
            cells_in_columns
        )

        self.assert_(np.all(next_active_cells < self.dchmm.total_cells))

    def test_prediction(self):
        self.dchmm.predict_cells()

    def test_grow_new_segments(self):
        new_segment_cells = (
                np.arange(self.dchmm.n_hidden_vars) * self.dchmm.n_hidden_states +
                self.dchmm._rng.choice(
                    np.arange(self.dchmm.n_hidden_states),
                    size=4,
                    replace=False
                )
        )

        growth_candidates = (
                np.arange(self.dchmm.n_hidden_vars) * self.dchmm.n_hidden_states +
                self.dchmm._rng.choice(
                    np.arange(self.dchmm.n_hidden_states),
                    size=4,
                    replace=False
                )
        )

        self.dchmm._grow_new_segments(
            new_segment_cells,
            growth_candidates
        )

    def test_calculate_learning_segments(self):
        prev_active_cells = (
                np.arange(self.dchmm.n_hidden_vars) * self.dchmm.n_hidden_states +
                self.dchmm._rng.choice(
                    np.arange(self.dchmm.n_hidden_states),
                    size=4,
                    replace=False
                )
        )

        next_active_cells = (
                np.arange(self.dchmm.n_hidden_vars) * self.dchmm.n_hidden_states +
                self.dchmm._rng.choice(
                    np.arange(self.dchmm.n_hidden_states),
                    size=4,
                    replace=False
                )
        )

        self.dchmm._calculate_learning_segments(prev_active_cells, next_active_cells)

    def test_update_factors(self):
        segments = self.dchmm._rng.choice(
            np.arange(self.dchmm.total_segments),
            size=4
        )
        segments_to_reinforce = segments[:2]
        segments_to_punish = segments[2:]

        self.dchmm._update_factors(segments_to_reinforce, segments_to_punish)

    def test_get_cells_for_observation(self):
        obs = np.arange(self.dchmm.n_obs_vars) * self.dchmm.n_obs_states + self.dchmm._rng.integers(
            low=0,
            high=self.dchmm.n_obs_states,
            size=self.dchmm.n_obs_vars
        )

        cells_in_columns = self.dchmm._get_cells_for_observation(obs)
