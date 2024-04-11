#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_digits

from hima.common.sdrr import RateSdr
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


class MnistDataset:
    images: npt.NDArray[float]
    target: npt.NDArray[int]

    dense_sdrs: npt.NDArray[float]
    sdrs: list[SparseSdr]
    classes: list[npt.NDArray[int]]

    output_sds: Sds
    binary: bool

    def __init__(self, binary: bool = True):
        self.binary = binary
        self.digits = load_digits()
        # normalize the images [0, 16] -> [0, 1]
        normalizer = 16.0
        self.images = self.digits.images / normalizer
        self.target = self.digits.target

        # Rate SDR
        self.dense_values = self.images.reshape(self.n_images, -1)
        threshold = 1.0 / normalizer
        self.sparse_sdrs = [np.flatnonzero(img >= threshold) for img in self.dense_values]

        # Binary SDR
        image_thresholds = np.mean(self.dense_values, axis=-1, keepdims=True)
        self.binary_dense_sdrs = (self.dense_values >= image_thresholds).astype(float)
        self.binary_sparse_sdrs = [np.flatnonzero(img) for img in self.binary_dense_sdrs]

        if binary:
            self.output_sds = Sds(shape=self.image_shape, sparsity=self.binary_dense_sdrs.mean())
        else:
            self.output_sds = Sds(shape=self.image_shape, sparsity=self.dense_values.mean())

        self.classes = [
            np.flatnonzero(self.target == cls)
            for cls in range(self.n_classes)
        ]

    def get_sdr(self, ind: int) -> SparseSdr | RateSdr:
        if self.binary:
            return self.binary_sparse_sdrs[ind]
        return RateSdr(self.sparse_sdrs[ind], values=self.dense_values[ind])

    @property
    def n_images(self):
        return self.images.shape[0]

    @property
    def n_classes(self):
        return 10

    @property
    def image_shape(self):
        return self.images.shape[1:]
