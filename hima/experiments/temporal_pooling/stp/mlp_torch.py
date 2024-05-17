#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn, optim

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


class MlpClassifier:
    feedforward_sds: Sds
    output_size: int

    # cache
    sparse_input: SparseSdr
    dense_input: npt.NDArray[float]
    sparse_gt: SparseSdr
    dense_gt_values: npt.NDArray[float]
    dense_gt_sdr: npt.NDArray[float]

    # learning stage tracking
    n_updates: int

    accumulated_loss: float | torch.Tensor
    accumulated_loss_steps: int | None
    loss_propagation_schedule: int

    def __init__(
            self,
            classification: bool,
            feedforward_sds: Sds, output_size: int,
            learning_rate: float,
            seed: int = None,
            collect_losses: int = 0,
            hidden_layer: bool | int = False
    ):
        self.rng = np.random.default_rng(seed)
        self.feedforward_sds = feedforward_sds
        self.output_size = output_size

        shape = (feedforward_sds.size, output_size)

        self.lr = learning_rate

        if hidden_layer:
            layers = [
                nn.Linear(shape[0], hidden_layer, dtype=float),
                nn.Tanh(),
                nn.Linear(hidden_layer, shape[1], dtype=float)
            ]
        else:
            layers = [
                nn.Linear(shape[0], shape[1], dtype=float),
            ]

        self.mlp = nn.Sequential(*layers)

        if classification:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.MSELoss()

        self.optim = optim.Adam(self.mlp.parameters(), lr=self.lr)

        self.collect_losses = collect_losses
        if self.collect_losses > 0:
            from collections import deque
            self.losses = deque(maxlen=self.collect_losses)

    def predict(self, dense_sdr: npt.NDArray[float]) -> npt.NDArray[float]:
        dense_sdr = torch.from_numpy(dense_sdr)
        with torch.no_grad():
            return self.mlp(dense_sdr).numpy()

    def learn(self, batch_dense_sdr: npt.NDArray[float], targets: npt.NDArray[int]):
        batch_dense_sdr = torch.from_numpy(batch_dense_sdr)
        targets = torch.from_numpy(targets)

        self.optim.zero_grad()

        loss = self.loss_func(self.mlp(batch_dense_sdr), targets)
        loss.backward()
        self.optim.step()

        if self.collect_losses:
            self.losses.append(loss.item())
