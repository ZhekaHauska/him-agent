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
from hima.experiments.temporal_pooling.stp.mlp_decoder_torch import SymExpModule


class MlpClassifier:
    layer_dims: list[int]

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
            classification: bool, layers: list[int],
            learning_rate: float,
            seed: int = None,
            collect_losses: int = 0,
            symexp_logits: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        self.lr = learning_rate

        self.layer_dims = layers

        nn_layers = [
            nn.Linear(layers[0], layers[1], dtype=float)
        ]
        for i in range(1, len(layers) - 1):
            nn_layers.append(nn.SiLU())
            nn_layers.append(nn.Linear(layers[i], layers[i + 1], dtype=float))

        if symexp_logits:
            nn_layers.append(SymExpModule())
        self.mlp = nn.Sequential(*nn_layers)

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

    @property
    def input_size(self):
        return self.layer_dims[0]

    @property
    def output_size(self):
        return self.layer_dims[-1]
