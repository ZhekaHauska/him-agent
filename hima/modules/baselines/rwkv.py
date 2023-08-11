#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import os
from rwkv.src.model import RWKV
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.autograd.set_detect_anomaly(True)


class RwkvWorldModelIterative:
    def __init__(
            self,
            n_obs_states,
            n_hidden_states,
            lr=2e-3,
            seed=None
    ):
        os.environ["RWKV_JIT_ON"] = '1'
        os.environ["RWKV_CUDA_ON"] = '0'

        self.n_obs_states = n_obs_states
        self.n_hidden_states = n_hidden_states
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm = RwkvWorldModelUnit(
            input_size=n_obs_states,
            hidden_size=n_hidden_states
        ).to(self.device)

        self.prediction = torch.zeros(self.n_obs_states, device=self.device)
        self.loss = None

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.RMSprop(self.lstm.parameters(), lr=self.lr)

        if seed is not None:
            torch.manual_seed(seed)

        self._rng = np.random.default_rng(seed)

    def observe(self, obs, learn=True):
        dense_obs = np.zeros(self.n_obs_states, dtype='float32')
        dense_obs[obs] = 1

        dense_obs = torch.from_numpy(dense_obs).to(self.device)

        if learn:
            if self.loss is None:
                self.loss = self.loss_function(self.prediction, dense_obs)
            else:
                self.loss += self.loss_function(self.prediction, dense_obs)

            self.prediction = self.lstm(dense_obs)
        else:
            with torch.no_grad():
                self.prediction = self.lstm(dense_obs)

    def reset(self):
        self.optimizer.zero_grad()

        if self.loss is not None:
            self.loss.backward()
            self.optimizer.step()
            self.loss = None

        self.prediction = torch.zeros(self.n_obs_states, device=self.device)
        self.lstm.message = (
            torch.zeros(self.n_hidden_states, device=self.device),
            torch.zeros(self.n_hidden_states, device=self.device),
        )

    def n_step_prediction(self, initial_dist, steps, mc_iterations=100):
        n_step_dist = np.zeros((steps, self.n_obs_states))
        initial_message = (self.lstm.message[0].clone(), self.lstm.message[1].clone())

        for i in range(mc_iterations):
            dist_curr_step = initial_dist
            for step in range(steps):
                # sample observation from prediction density
                gamma = self._rng.random(size=self.n_obs_states)
                obs = np.flatnonzero(gamma < dist_curr_step)
                dense_obs = np.zeros(self.n_obs_states, dtype='float32')
                dense_obs[obs] = 1
                dense_obs = torch.from_numpy(dense_obs).to(self.device)

                # predict distribution
                with torch.no_grad():
                    prediction = self.lstm(dense_obs).cpu().detach().numpy()

                n_step_dist[step] += 1/(i+1) * (prediction - n_step_dist[step])
                dist_curr_step = prediction

            self.lstm.message = initial_message

        return n_step_dist


class RwkvWorldModelUnit(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size
    ):
        super(RwkvWorldModelUnit, self).__init__()

        self.n_obs_states = input_size
        self.n_hidden_states = hidden_size

        self.lstm = nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size
        )

        # The linear layer that maps from hidden state space back to obs space
        self.hidden2obs = nn.Linear(
            hidden_size,
            input_size
        )

        self.message = (
            torch.zeros(self.n_hidden_states),
            torch.zeros(self.n_hidden_states)
        )

    def forward(self, obs):
        self.message = self.lstm(obs, self.message)
        prediction_logit = self.hidden2obs(self.message[0])
        prediction = torch.sigmoid(prediction_logit)
        return prediction

