#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import unittest
from hima.envs.pixel.pixball import Pixball
import imageio
import yaml


class PixballTest(unittest.TestCase):
    def setUp(self) -> None:
        with open('configs/pixball.yaml', 'r') as file:
            config = yaml.load(file, Loader=yaml.Loader)

        self.env = Pixball(**config)

    def test_basic(self):
        self.env.reset()
        self.env.act([1, 1])

        with imageio.get_writer(
            f'test.gif',
            mode='I',
            fps=10
        ) as writer:
            for i in range(50):
                im = self.env.obs()
                writer.append_data(im * 255)

                self.env.step()

