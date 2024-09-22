#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
import torch
from torch.nn import functional as F
from torchvae.cat_vae import CategoricalVAE
from torchvision.transforms import ToTensor
from hima.modules.belief.cortial_column.encoders.base import BaseEncoder


class CatVAE(BaseEncoder):
    def __init__(self, checkpoint_path, use_camera, model_params):
        self.use_camera = use_camera
        self.model = CategoricalVAE(**model_params)
        self.input_shape = (64, 64)
        state_dict = torch.load(checkpoint_path)['state_dict']
        state_dict = {'.'.join(key.split('.')[1:]): value for key, value in state_dict.items()}

        self.transform = ToTensor()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.n_states = self.model.categorical_dim
        self.n_vars = self.model.latent_dim

    def encode(self, input_: np.ndarray, learn: bool) -> np.ndarray:
        if self.use_camera:
            pic = np.zeros(np.prod(self.input_shape), dtype=np.float32)
            pic[input_] = 1
            pic = pic.reshape(self.input_shape)
        else:
            pic = input_.astype(np.float32)

        input_ = self.transform(pic)
        input_ = input_.unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.model.encode(input_)[0]
            dense = F.softmax(z / self.model.temp, dim=-1)
        dense = dense.squeeze(0).view(self.n_vars, self.n_states)
        dense = dense.detach().cpu().numpy()
        result = (
                np.argmax(dense, axis=-1) +
                np.arange(self.n_vars) * self.n_states
        )
        return result
