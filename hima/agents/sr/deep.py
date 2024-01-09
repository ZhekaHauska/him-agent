#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.modules.baselines.srtd import SRTD
from hima.modules.baselines.lstm import to_numpy
import numpy as np
import torch


class DeepSR:
    def __init__(
            self,
            hidden_state_size,
            gamma,
            rew_lr,
            lr,
            tau,
            batch_size,
            hidden_size,
            n_hidden_layers
    ):
        self.rew_lr = rew_lr
        self.gamma = gamma
        self.hidden_state_size = hidden_state_size
        self.srtd = SRTD(
            self.hidden_state_size,
            self.hidden_state_size,
            lr=lr,
            tau=tau,
            batch_size=batch_size,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers
        )

        self.hidden_state = None
        self.prev_hidden_state = None
        hidden_state_rewards = np.zeros(self.hidden_state_size)
        self.hidden_state_rewards = hidden_state_rewards.flatten()

    def observe(self, hidden_state, learn=True):
        self.prev_hidden_state = self.hidden_state.copy()
        self.hidden_state = hidden_state

        if learn:
            self.td_update_sr()

    def sample_action(self):
        ...

    def reinforce(self, reward):
        self.hidden_state_rewards += self.rew_lr * self.hidden_state * (
                reward - self.hidden_state_rewards
        )

    def td_update_sr(self):
        prev_obs = torch.tensor(self.prev_hidden_state).float().to(self.srtd.device)
        predicted_sr = self.srtd.predict_sr(prev_obs, target=False)
        target_sr = self.hidden_state + self.gamma * predicted_sr

        target_sr = torch.tensor(target_sr)
        target_sr = target_sr.float().to(self.srtd.device)

        td_error = self.srtd.compute_td_loss(
            target_sr,
            predicted_sr
        )

        return to_numpy(predicted_sr), to_numpy(target_sr), td_error

    def predict_sr(self, observation):
        msg = torch.tensor(observation).float().to(self.srtd.device)
        return to_numpy(self.srtd.predict_sr(msg, target=True))
