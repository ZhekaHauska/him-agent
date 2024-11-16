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
from dhtm.modules.belief.cortial_column.encoders.base import BaseEncoder


class CatVAE(BaseEncoder):
    def __init__(
            self,
            checkpoint_path,
            use_camera,
            model_params,
            force_one_hot=False,
            max_vocab_size=1000
    ):
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

        self.force_one_hot = force_one_hot
        self.max_vocab_size = max_vocab_size
        self.vocab = dict()
        self.vocab_size = 0

        if self.force_one_hot:
            self.n_states = self.max_vocab_size
            self.n_vars = 1
        else:
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
        dense = dense.squeeze(0).view(self.model.latent_dim, self.model.categorical_dim)
        dense = dense.detach().cpu().numpy()
        sparse = (
                np.argmax(dense, axis=-1) +
                np.arange(self.model.latent_dim) * self.model.categorical_dim
        )

        if self.force_one_hot:
            sparse = str(sparse)
            if sparse in self.vocab:
                result = self.vocab[sparse]
            else:
                if self.vocab_size < self.max_vocab_size:
                    self.vocab[sparse] = self.vocab_size
                    result = self.vocab_size
                    self.vocab_size += 1
                else:
                    # randomly replace old key
                    result = self.vocab.pop(next(iter(self.vocab.keys())))
                    self.vocab[sparse] = result
            result = [result]
        else:
            result = sparse

        return result

    @property
    def dict_size(self):
        return len(self.vocab)
