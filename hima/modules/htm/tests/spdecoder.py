#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.modules.htm.spatial_pooler import SPDecoder, HtmSpatialPooler, SPEnsemble
from htm.bindings.sdr import SDR
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from unittest import TestCase
import yaml


class TestSPEncoderDecoder(TestCase):
    def setUp(self) -> None:
        with open('configs/sp_default.yaml', 'r') as file:
            self.config = yaml.load(file, Loader=yaml.Loader)

        if 'n_sp' in self.config:
            self.sp = SPEnsemble(
                **self.config
            )
        else:
            self.sp = HtmSpatialPooler(
                **self.config
            )

        self.decoder = SPDecoder(self.sp)

    def test_encoder(self):
        input_sdr = SDR(self.sp.getNumInputs())
        output_sdr = SDR(self.sp.getNumColumns())

        input_sdr.sparse = np.random.randint(input_sdr.size, size=2)
        self.sp.compute(input_sdr, True, output_sdr)

        print(input_sdr.sparse)
        print(output_sdr.sparse)

        self.sp.getColumnDimensions()

    def test_decoder(self):
        probs = np.random.rand(self.sp.getNumColumns())

        res = self.decoder.decode(probs, update=True)

        sns.heatmap(res.reshape(self.sp.getInputDimensions()))
        plt.show()
