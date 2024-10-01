#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.modules.baselines.lstm import LSTMWMUnit
import torch
import numpy as np


if __name__ == '__main__':
    device = 'cpu'
    lstm = LSTMWMUnit(
        8, 10
    ).to(device)

    prediction = torch.zeros(8, device=device)
    observations = np.random.randint(8, size=10)
    loss_function = torch.nn.BCELoss()

    for obs in observations:
        dense_obs = np.zeros(8, dtype='float32')
        dense_obs[obs] = 1
        dense_obs = torch.from_numpy(dense_obs).to(device)

        loss = loss_function(prediction, dense_obs)
        if loss.requires_grad:
            loss.backward(retain_graph=True)

        prediction = lstm(dense_obs)
