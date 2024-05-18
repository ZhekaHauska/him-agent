#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
import torch
from torchvae.cat_vae import CategoricalVAE
from torchvision.transforms import ToTensor
from hima.common.sdr import sparse_to_dense
from htm.bindings.sdr import SDR


class CatVAE:
    def __init__(self, checkpoint_path, model_params):
        self.model = CategoricalVAE(**model_params)
        self.input_shape = (64, 64)
        state_dict = torch.load(checkpoint_path)['state_dict']
        state_dict = {'.'.join(key.split('.')[1:]): value for key, value in state_dict.items()}

        self.transform = ToTensor()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute(self, input_sdr: SDR, learn: bool, output_sdr: SDR = None):
        if isinstance(input_sdr, SDR):
            input_sdr = input_sdr.dense.copy().reshape(self.input_shape)
        else:
            input_sdr = sparse_to_dense(input_sdr, shape=self.input_shape)

        input_sdr = input_sdr.astype(np.float32)
        input_sdr = self.transform(input_sdr)
        input_sdr = input_sdr.unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.model.encode(input_sdr)[0]
            dense = self.model.reparameterize(z)
        dense = dense.squeeze(0).view(self.model.latent_dim, self.model.categorical_dim)
        dense = dense.detach().cpu().numpy()
        result = (
                np.argmax(dense, axis=-1) +
                np.arange(self.model.latent_dim) * self.model.categorical_dim
        )

        if output_sdr is not None:
            output_sdr.sparse = result
        return result

    def getSingleNumColumns(self):
        return self.model.categorical_dim

    def getNumColumns(self):
        return self.model.categorical_dim * self.model.latent_dim

    def getInputDimensions(self):
        return self.input_shape

    def getNumInputs(self):
        return np.prod(self.input_shape)
