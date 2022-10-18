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

    def test_sampler(self):
        probs = self.dchmm._rng.dirichlet(
                    alpha=[1.0]*self.dchmm.n_hidden_states,
                    size=self.dchmm.n_hidden_vars
                )

        sample = self.dchmm._sample_categorical_variables(probs)

        self.assertTupleEqual(sample.shape, (self.dchmm.n_hidden_vars,))
        self.assert_(np.all((sample < self.dchmm.n_hidden_states) & (sample >= 0)))

    def test_learning(self):
        self.dchmm.prediction = self.dchmm._rng.dirichlet(
                    alpha=[1.0]*self.dchmm.n_hidden_states,
                    size=self.dchmm.n_hidden_vars
                ).flatten()

        obs = self.dchmm._rng.integers(
            low=0,
            high=self.dchmm.n_obs_states,
            size=self.dchmm.n_obs_vars
        )

        self.dchmm.observe(obs, learn=True)

