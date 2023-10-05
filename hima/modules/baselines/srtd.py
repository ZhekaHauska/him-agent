#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            activation_function
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        all_sizes = [input_size]
        all_sizes.extend(hidden_sizes)
        all_sizes.append(output_size)

        stack = list()
        for i, h_size in enumerate(hidden_sizes):
            stack.append(nn.Linear(all_sizes[i], all_sizes[i+1]))
            stack.append(activation_function())
        # output layer
        stack.append(nn.Linear(all_sizes[-2], all_sizes[-1]))

        self.fcnn = nn.Sequential(
            *stack
        )

    def forward(self, x):
        return self.fcnn(x)


class SRTD:
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size=256,
            n_hidden_layers=1,
            activation_function=nn.SiLU,
            lr=0.01,
            tau=0.01,
            batch_size=32
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = lr
        self.tau = tau  # target soft update rate
        self.batch_size = batch_size
        self.sample_counter = 0

        self.model = MLP(
            input_size,
            output_size,
            [hidden_size]*n_hidden_layers,
            activation_function
        ).to(self.device)
        self.model_target = MLP(
            input_size,
            output_size,
            [hidden_size]*n_hidden_layers,
            activation_function
        ).to(self.device)
        self.model_target.load_state_dict(self.model.state_dict())

        self.accumulated_td_loss = None
        self.mse = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)

    def predict_sr(self, state, target=True):
        if target:
            with torch.no_grad():
                predicted_sr = self.model_target(state).detach()
        else:
            predicted_sr = self.model(state)
        return predicted_sr

    def compute_td_loss(self, target_sr, predicted_sr):
        td_loss = self.mse(target_sr, predicted_sr)

        if self.accumulated_td_loss is None:
            self.accumulated_td_loss = 0
        self.accumulated_td_loss += td_loss

        self.sample_counter += 1
        if self.sample_counter >= self.batch_size:
            self.update()

        return td_loss.item()

    def update(self):
        if (self.accumulated_td_loss is None) or (self.sample_counter == 0):
            return

        self.optimizer.zero_grad()
        mean_loss = self.accumulated_td_loss / self.sample_counter
        mean_loss.backward()
        self.optimizer.step()

        self.accumulated_td_loss = None
        self.sample_counter = 0

        # soft target update
        model_target_state_dict = self.model_target.state_dict()
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            model_target_state_dict[key] = (
                    model_state_dict[key] * self.tau +
                    model_target_state_dict[key] * (1 - self.tau)
            )
        self.model_target.load_state_dict(model_target_state_dict)
