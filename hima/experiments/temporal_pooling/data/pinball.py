#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

from hima.common.sds import Sds


class PinballDataset:
    images: np.ndarray
    target: np.ndarray

    def __init__(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        self.images = np.load(path)
        self.dense_sdrs = self.images.reshape(self.n_images, -1)
        self.sdrs = [np.flatnonzero(img) for img in self.dense_sdrs]
        self.output_sds = Sds(size=self.dense_sdrs.shape[-1], sparsity=self.dense_sdrs.mean())

    @property
    def info(self) -> str:
        return f'#{self.n_images}: {self.image_shape} | {self.output_sds}'

    @property
    def n_images(self):
        return self.images.shape[0]

    @property
    def image_shape(self):
        return self.images.shape[1:]
