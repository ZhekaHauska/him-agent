#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits

from hima.common.float_sdr import FloatSparseSdr
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


class MnistDataset:
    images: np.ndarray
    target: np.ndarray

    dense_sdrs: np.ndarray
    sdrs: list[SparseSdr]
    classes: list[np.ndarray]

    output_sds: Sds

    def __init__(self):
        self.digits = load_digits()
        self.images = self.digits.images
        self.target = self.digits.target

        flatten_images: np.ndarray = self.images.reshape(self.n_images, -1)
        image_thresholds = np.mean(flatten_images, axis=-1, keepdims=True)
        # noinspection PyUnresolvedReferences
        self.dense_sdrs = (flatten_images >= image_thresholds).astype(float)
        self.sdrs = [np.flatnonzero(img) for img in self.dense_sdrs]
        self.sparse_values = [flatten_images[i][self.sdrs[i]] for i in range(self.n_images)]
        self.output_sds = Sds(shape=self.image_shape, sparsity=self.dense_sdrs.mean())

        self.classes = [
            np.flatnonzero(self.target == cls)
            for cls in range(self.n_classes)
        ]

    def get_sdr(self, ind: int, binary: bool = True) -> SparseSdr | FloatSparseSdr:
        if binary:
            return self.sdrs[ind]
        return FloatSparseSdr(self.sdrs[ind], values=self.sparse_values[ind])

    @property
    def n_images(self):
        return self.images.shape[0]

    @property
    def n_classes(self):
        return 10

    @property
    def image_shape(self):
        return self.images.shape[1:]
