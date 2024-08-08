#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hima.common.sdr import SparseSdr, RateSdr
from hima.common.sdr_array import SdrArray
from hima.common.sds import Sds


@dataclass
class SdrDataset:
    binary: bool
    sdrs: SdrArray

    # raw 2D images
    images: npt.NDArray[float]
    # class indices
    targets: npt.NDArray[int]

    _classes: list[npt.NDArray[int]] | None

    def __init__(self, images, targets, threshold: float, binary: bool):
        self.binary = binary
        self.images = images
        self.targets = targets

        flatten_images = self.images.reshape(self.images.shape[0], -1).copy()
        sdr_size = flatten_images.shape[1]

        if binary:
            bin_rate_sdrs = [
                RateSdr(np.flatnonzero(img >= img.mean()))
                for img in flatten_images
            ]
            self.sdrs = SdrArray(sparse=bin_rate_sdrs, sdr_size=sdr_size)
        else:
            bin_sdrs = [np.flatnonzero(img >= threshold) for img in flatten_images]
            flatten_images[flatten_images < threshold] = 0.0
            rate_sdrs = [
                RateSdr(sdr, values=values[sdr])
                for sdr, values in zip(bin_sdrs, flatten_images)
            ]
            self.sdrs = SdrArray(sparse=rate_sdrs, dense=flatten_images, sdr_size=sdr_size)
        self._classes = None

    def __len__(self):
        return len(self.sdrs)

    @property
    def n_classes(self):
        return 10

    @property
    def classes(self):
        if self._classes is None:
            self._classes = [np.flatnonzero(self.targets == i) for i in range(self.n_classes)]
        return self._classes

    @property
    def image_shape(self):
        return self.images.shape[1:]

    def get_sdr(self, ind: int) -> SparseSdr | RateSdr:
        return self.sdrs.get_sdr(ind, binary=self.binary)

    def normalize(self, normalizer):
        self.sdrs = self.sdrs.create_modified(normalizer)


class MnistDataset:
    images: npt.NDArray[float]
    targets: npt.NDArray[int]

    train: SdrDataset
    test: SdrDataset

    sds: Sds
    binary: bool

    def __init__(self, seed: int, binary: bool = True, ds: str = 'mnist', debug: bool = False):
        self.binary = binary
        threshold, train, test = _load_dataset(seed, ds, grayscale=True, debug=debug)

        train_images, train_targets = train
        self.train = SdrDataset(train_images, train_targets, threshold, binary)

        test_images, test_targets = test
        self.test = SdrDataset(test_images, test_targets, threshold, binary)

        sum_active = np.sum([len(rate_sdr.sdr) for rate_sdr in self.train.sdrs.sparse])
        total_number = self.train.images.size
        sparsity = sum_active / total_number
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
    threshold = 2.0 / normalizer

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

    return threshold, (train_images, train_targets), (test_images, test_targets)
