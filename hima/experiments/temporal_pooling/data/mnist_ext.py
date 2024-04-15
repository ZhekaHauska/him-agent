#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hima.common.sdrr import RateSdr
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


@dataclass
class SdrDataset:
    binary: bool

    images: npt.NDArray[float]
    targets: npt.NDArray[int]

    dense_sdrs: npt.NDArray[float]
    sdrs: list[SparseSdr]
    classes: list[npt.NDArray[int]]

    def __init__(self, images, targets, threshold: float, binary: bool):
        self.binary = binary

        self.images = images
        self.targets = targets

        # Rate SDR
        self.dense_values = self.images.reshape(self.n_images, -1)
        self.sparse_sdrs = [np.flatnonzero(img >= threshold) for img in self.dense_values]

        # Binary SDR
        image_thresholds = np.mean(self.dense_values, axis=-1, keepdims=True)
        self.binary_dense_sdrs = (self.dense_values >= image_thresholds).astype(float)
        self.binary_sparse_sdrs = [np.flatnonzero(img) for img in self.binary_dense_sdrs]

        self.classes = [
            np.flatnonzero(self.targets == cls)
            for cls in range(self.n_classes)
        ]

    @property
    def n_images(self):
        return self.images.shape[0]

    @property
    def n_classes(self):
        return 10

    @property
    def image_shape(self):
        return self.images.shape[1:]

    def get_sdr(self, ind: int) -> SparseSdr | RateSdr:
        if self.binary:
            return self.binary_sparse_sdrs[ind]
        return RateSdr(self.sparse_sdrs[ind], values=self.dense_values[ind])


class MnistDataset:
    images: npt.NDArray[float]
    targets: npt.NDArray[int]

    train: SdrDataset
    test: SdrDataset

    output_sds: Sds
    binary: bool

    def __init__(self, seed: int, binary: bool = True):
        self.binary = binary
        normalizer, train, test = load_mnist(seed)

        # NB: to get sdr for rate sdrs
        threshold = 1.0 / normalizer

        train_images, train_targets = train
        self.train = SdrDataset(train_images, train_targets, threshold, binary)

        test_images, test_targets = test
        self.test = SdrDataset(test_images, test_targets, threshold, binary)

        sparsity = self.train.binary_dense_sdrs.mean() if binary else self.train.dense_values.mean()
        self.output_sds = Sds(shape=self.image_shape, sparsity=sparsity)

    @property
    def n_classes(self):
        return 10

    @property
    def image_shape(self):
        return self.train.images.shape[1:]


def load_mnist(seed: int, test_size: int | float = 10_000):
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    images, targets = fetch_openml(
        'mnist_784', version=1, return_X_y=True, as_frame=False,
        parser='auto'
    )
    print(f'MNIST LOADED images: {images.shape} | targets: {targets.shape}')

    # normalize the images [0, 255] -> [0, 1]
    normalizer = 255.0
    images = images.astype(float) / normalizer
    targets = targets.astype(int)

    train_images, test_images, train_targets, test_targets = train_test_split(
        images, targets, random_state=seed, test_size=test_size
    )
    return normalizer, (train_images, train_targets), (test_images, test_targets)
