#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hima.common.sdr import sparse_to_dense, SparseSdr, DenseSdr
from hima.common.utils import safe_divide


THiddenState = tuple[torch.Tensor, torch.Tensor]


class LstmLayer:
    predicted_observation: torch.Tensor | None
    predicted_observation_npy: DenseSdr | None

    def __init__(
            self, *,
            input_size: int,
            hidden_size: int,
            lr=2e-3,
            seed=None,
            decoder_bias: bool = True,
    ):
        torch.set_num_threads(1)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm = LstmUnit(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            decoder_bias=decoder_bias
        ).to(self.device)

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.RMSprop(self.lstm.parameters(), lr=self.lr)

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.lstm.message = self.lstm.get_init_message()
        self.predicted_observation = None
        self.predicted_observation_npy = None
        self.make_stats(
            [],
            np.zeros(self.input_size, dtype=int)
        )

    def observe(self, observation: SparseSdr, learn: bool = True):
        dense_obs = sparse_to_dense(observation, size=self.input_size)
        dense_obs = torch.from_numpy(dense_obs).float().to(self.device)

        correctly_predicted = []
        if learn and self.predicted_observation is not None:
            self.optimizer.zero_grad()
            loss = self.loss_function(self.predicted_observation, dense_obs)
            loss.backward()
            self.optimizer.step()

            self.lstm.message = (
                self.lstm.message[0].detach(),
                self.lstm.message[1].detach(),
            )

            correctly_predicted = np.flatnonzero(self.predicted_observation[observation] > 0.5)
            self.make_stats(observation, dense_obs)

        with torch.set_grad_enabled(learn):
            self.lstm.transition_to_next_state(dense_obs)
            self.predicted_observation = self.lstm.decode_obs()

        self.predicted_observation_npy = to_numpy(self.predicted_observation)
        return self.predicted_observation_npy, correctly_predicted

    def make_stats(self, obs_sdr: SparseSdr, dense_obs: DenseSdr):
        tp = 0.
        tp_fp = 0.
        if self.predicted_observation_npy is not None:
            tp = np.dot(dense_obs, self.predicted_observation_npy)
            tp_fp = self.predicted_observation_npy.sum()
        tp_fn = len(obs_sdr)

        # precision: tp / tp+fp
        precision = safe_divide(tp, tp_fp)
        # recall: tp / tp+fn
        recall = safe_divide(tp, tp_fn)

        # anomaly, miss rate: 1 - recall = fp / tp+fp
        self.column_miss_rate = 1 - recall
        self.anomaly = [self.column_miss_rate]
        # precision
        self.column_precision = precision
        self.column_imprecision = 1 - self.column_precision
        # predicted / actual == prediction relative sparsity
        self.column_prediction_volume = safe_divide(tp_fp, tp_fn)

        self.cell_imprecision = self.column_imprecision
        # predicted / actual == prediction relative sparsity
        self.cell_prediction_volume = self.column_prediction_volume


class LstmUnit(nn.Module):
    def __init__(
            self, *,
            input_size,
            hidden_size,
            decoder_bias: bool = True
    ):
        super(LstmUnit, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_obs_states = input_size
        self.n_hidden_states = hidden_size

        self.lstm = nn.LSTMCell(
            input_size=self.n_obs_states,
            hidden_size=self.n_hidden_states
        )

        # The linear layer that maps from hidden state space back to obs space
        self.hidden2obs = nn.Linear(
            self.n_hidden_states,
            self.n_obs_states,
            bias=decoder_bias
        )

        self.message = self.get_init_message()

    def get_init_message(self) -> THiddenState:
        return (
            torch.zeros(self.n_hidden_states, device=self.device),
            torch.zeros(self.n_hidden_states, device=self.device)
        )

    def transition_to_next_state(self, obs):
        self.message = self.lstm(obs, self.message)
        return self.message

    def decode_obs(self):
        obs_msg = self.message[0]
        prediction_logit = self.hidden2obs(obs_msg)
        prediction = torch.sigmoid(prediction_logit)
        return prediction

    def forward(self, obs):
        self.transition_to_next_state(obs)
        return self.decode_obs()


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    # torch
    return x.detach().cpu().numpy()
