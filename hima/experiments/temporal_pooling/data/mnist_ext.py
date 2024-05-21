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
        image_thresholds = np.maximum(image_thresholds, threshold)
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

        sdr = self.sparse_sdrs[ind]
        return RateSdr(sdr, values=self.dense_values[ind][sdr])


class MnistDataset:
    images: npt.NDArray[float]
    targets: npt.NDArray[int]

    train: SdrDataset
    test: SdrDataset

    sds: Sds
    binary: bool

    def __init__(self, seed: int, binary: bool = True, ds: str = 'mnist', debug: bool = False):
        self.binary = binary
        normalizer, threshold, train, test = _load_dataset(seed, ds, grayscale=True, debug=debug)

        train_images, train_targets = train
        self.train = SdrDataset(train_images, train_targets, threshold, binary)

        test_images, test_targets = test
        self.test = SdrDataset(test_images, test_targets, threshold, binary)

        sparsity = self.train.binary_dense_sdrs.mean() if binary else self.train.dense_values.mean()
        self.sds = Sds(shape=self.image_shape, sparsity=sparsity)

    @property
    def n_classes(self):
        return 10

    @property
    def image_shape(self):
        return self.train.images.shape[1:]


def _load_dataset(
        seed: int, ds_name: str, test_size: int | float = 10_000, grayscale: bool = True,
        debug: bool = False
):
    # normalize the images [0, 255] -> [0, 1]
    normalizer = 255.0

    # NB: to get sdr for rate sdrs
    if ds_name == 'mnist':
        threshold = 0.05
    else:
        threshold = 0.15

    from pathlib import Path
    cache_path = Path(f'~/data/_cache/{ds_name}{"_gs" if grayscale else ""}.pkl')
    cache_path = cache_path.expanduser()

    if cache_path.exists():
        import pickle
        with cache_path.open('rb') as f:
            ds = pickle.load(f)
            images, targets = ds['images'], ds['targets']
    else:
        from sklearn.datasets import fetch_openml

        supported_datasets = {'mnist': 'mnist_784', 'cifar': 'cifar_10'}
        images, targets = fetch_openml(
            supported_datasets[ds_name], version=1, return_X_y=True, as_frame=False,
            parser='auto'
        )

        images = images.astype(float) / normalizer
        if grayscale and ds_name == 'cifar':
            # convert to grayscale
            print('CONVERTING CIFAR TO GRAYSCALE')
            images = images[:, :1024] * 0.30 + images[:, 1024:2048] * 0.59 + images[:, 2048:] * 0.11

        targets = targets.astype(int)
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open('wb') as f:
            pickle.dump({'images': images, 'targets': targets}, f)

    print(f'{ds_name} LOADED images: {images.shape} | targets: {targets.shape}')

    from sklearn.model_selection import train_test_split
    train_images, test_images, train_targets, test_targets = train_test_split(
        images, targets, random_state=seed, test_size=test_size
    )

    # NB: remove after debug session
    if debug:
        n_trains, n_tests = 15_000, 2_500
        train_images, train_targets = train_images[:n_trains], train_targets[:n_trains]
        test_images, test_targets = test_images[:n_tests], test_targets[:n_tests]

    return normalizer, threshold, (train_images, train_targets), (test_images, test_targets)
