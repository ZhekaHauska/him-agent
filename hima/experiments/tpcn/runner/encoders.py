import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hima.modules.tpcn.utils import Tanh, Softmax
from hima.modules.tpcn.constants import ACTIVATION_FUNCS
from typing import Union, List, Literal
from numpy.typing import NDArray

INI_MODE = Literal['dirichlet', 'normal', 'uniform']
    
class SimpleOneHotEncoder:
    def __init__(self, max_categories: int):
        
        self.max_categories = max_categories
        self.categories = {}
    
    def fit(self, x: Union[NDArray[Union[np.int32, np.int16]], List[int]], int) -> None:
        
        if isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, int):
            x = np.array([x])
        
        unique_values = np.unique(x)

        for value in unique_values:
            self.categories[int(value)] = len(self.categories)
        
        return self

    def encode(self, x: Union[NDArray[Union[np.int32, np.int16]], List[int]]) -> NDArray:
        
        if isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, int):
            x = np.array([x])
        
        n_categories = np.unique(x).shape[0]
    
        if n_categories > self.max_categories:
            raise RuntimeError(f'The number of unique observations' \
                            f'{n_categories} is greater than the maximum allowed {self.max_categories}')

        result = np.zeros((x.shape[0], self.max_categories), dtype=np.float32)

        for i, value in enumerate(x):

            if int(value) not in self.categories.keys():
                self.categories[int(value)] = len(self.categories)
            
            result[i, self.categories[value]] = 1.0
        
        return result

class HierarchicalPCN(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 n_obs_states: int,
                 lambda_z_init: float,
                 need_onehot: bool,
                 inf_iters: int,
                 inf_lr: float,
                 out_activation: Literal['relu', 'tanh', 'sigmoid', 'softmax'],
                 loss: Literal['CE', 'BCE'] | None):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_obs_states = n_obs_states
        self.Wout = nn.Linear(hidden_size, n_obs_states, bias=False)
        self.mu = nn.Parameter(torch.zeros((hidden_size)))
        # sparse penalty
        self.inf_iters = inf_iters
        self.inf_lr = inf_lr
        self.sparse_z = lambda_z_init

        self.out_activation = ACTIVATION_FUNCS[out_activation]
        self.loss = loss

        if need_onehot:
            self.onehot_encoder = SimpleOneHotEncoder(max_categories=self.n_obs_states)

    def set_sparsity(self, sparsity):
        self.sparse_z = sparsity

    def set_nodes(self, inp):
        # intialize the value nodes
        self.z = self.mu.clone()
        self.x = inp.clone()

        # computing error nodes
        self.update_err_nodes()

    def decode(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).reshape(-1, self.hidden_size)
        return self.out_activation(self.Wout(z))

    def update_err_nodes(self):
        self.err_z = self.z - self.mu
        pred_x = self.decode(self.z)
        if isinstance(self.out_activation, Tanh):
            self.err_x = self.x - pred_x
        elif isinstance(self.out_activation, Softmax):
            self.err_x = self.x / (pred_x + 1e-8)
        else:
            self.err_x = self.x / (pred_x + 1e-8) + (1 - self.x) / (1 - pred_x + 1e-8)

    def inference_step(self, inf_lr):
        Wout = self.Wout.weight.clone().detach()
        if isinstance(self.out_activation, Softmax):
            delta = (
                self.err_z
                - (
                    self.out_activation.deriv(self.Wout(self.z))
                    @ self.err_x.unsqueeze(-1)
                ).squeeze(-1)
                @ Wout
            )
        else:
            delta = (
                self.err_z
                - (self.out_activation.deriv(self.Wout(self.z)) * self.err_x) @ Wout
            )
        delta += self.sparse_z * torch.sign(self.z)
        self.z = self.z - inf_lr * delta

    def inference(self, inf_iters, inf_lr, inp):
        self.set_nodes(inp)
        for itr in range(inf_iters):
            with torch.no_grad():
                self.inference_step(inf_lr)
            self.update_err_nodes()

    def get_energy(self):
        """Function to obtain the sum of all layers' squared MSE"""
        if self.loss == "CE":
            obs_loss = F.cross_entropy(self.Wout(self.z), self.x)
        elif self.loss == "BCE":
            obs_loss = F.binary_cross_entropy_with_logits(self.Wout(self.z), self.x)
        else:
            obs_loss = torch.sum(self.err_x**2, -1).mean()
        latent_loss = torch.sum(self.err_z**2, -1).mean()
        energy = obs_loss + latent_loss
        return energy, obs_loss

    def encode(self, x):
        self.eval()

        encoded_x = torch.tensor(self.onehot_encoder.encode(x))
        with torch.no_grad():
            self.inference(inf_iters=self.inf_iters, inf_lr=self.inf_lr, inp=encoded_x)
        
        return self.z.clone().detach().numpy(), encoded_x
            