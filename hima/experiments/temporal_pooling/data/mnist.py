#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from sklearn.datasets import load_digits

from hima.common.sds import Sds


class MnistDataset:
    images: np.ndarray
    target: np.ndarray

    def __init__(self):
        self.digits = load_digits()
        self.images = self.digits.images
        self.target = self.digits.target

        flatten_images: np.ndarray = self.images.reshape(self.n_images, -1)
        image_thresholds = np.mean(flatten_images, axis=-1, keepdims=True)
        self.dense_sdrs = flatten_images >= image_thresholds
        self.sdrs = [np.flatnonzero(img) for img in self.dense_sdrs]
        self.output_sds = Sds(size=self.dense_sdrs.shape[-1], sparsity=self.dense_sdrs.mean())

        self.classes = [
            np.flatnonzero(self.target == cls)
            for cls in range(10)
        ]

    @property
    def n_images(self):
        return self.images.shape[0]

    @property
    def image_shape(self):
        return self.images.shape[1:]
