#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import unittest
from hima.envs.mpg.mpg import draw_mpg, MarkovProcessGrammar
import yaml


class MPGTest(unittest.TestCase):
    def setUp(self) -> None:
        with open('configs/mpg_default.yaml', 'r') as file:
            config = yaml.load(file, Loader=yaml.Loader)

        self.mpg = MarkovProcessGrammar(**config)

    def test_visualization(self):
        draw_mpg(
            'a.png',
            self.mpg.transition_probs,
            self.mpg.transition_letters
        )
