import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from hima.modules.tpcn.utils import Tanh, Softmax
from hima.modules.tpcn.constants import ACTIVATION_FUNCS
from typing import Literal

class TemporalPCN(nn.Module):
    """Multi-layer tPC class, using autograd"""

    def __init__(self, 
                 hidden_size: int,
                 n_obs_states: int,
                 n_actions: int,
                 lambda_z: float,
                 weight_decay: float,
                 rec_activation: Literal['relu', 'tanh', 'sigmoid', 'softmax'],
                 out_activation: Literal['relu', 'tanh', 'sigmoid', 'softmax'],
                 loss: Literal['CE', 'BCE'] | None):
        super(TemporalPCN, self).__init__()

        self.name = 'tpcn'
        self.hidden_size = hidden_size
        self.n_obs_states = n_obs_states
        self.n_actions = n_actions
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Win = nn.Linear(n_actions, hidden_size, bias=False)
        self.Wout = nn.Linear(hidden_size, n_obs_states, bias=False)

        self.sparse_z = lambda_z
        self.weight_decay = weight_decay

        self.out_activation = ACTIVATION_FUNCS[out_activation]
        self.rec_activation = ACTIVATION_FUNCS[rec_activation]
        self.loss = loss

    def set_nodes(self, v, prev_z, p):
        """Set the initial value of the nodes;

        In particular, we initialize the hiddden state with a forward pass.

        Args:
            v: velocity input at a particular timestep in stimulus
            prev_z: previous hidden state
            p: place cell activity at a particular timestep in stimulus
        """
        self.z = self.g(v, prev_z)
        self.x = p.clone()
        self.update_err_nodes(v, prev_z)

    def update_err_nodes(self, v, prev_z):
        self.err_z = self.z - self.g(v, prev_z)
        pred_x = self.decode(self.z)
        if isinstance(self.out_activation, Tanh):
            self.err_x = self.x - pred_x
        elif isinstance(self.out_activation, Softmax):
            self.err_x = self.x / (pred_x + 1e-9)
        else:
            self.err_x = self.x / (pred_x + 1e-9) - (1 - self.x) / (1 - pred_x + 1e-9)

    def g(self, v, prev_z):
        return self.rec_activation(self.Wr(prev_z) + self.Win(v))

    def decode(self, z):
        return self.out_activation(self.Wout(z))

    def inference_step(self, inf_lr, v, prev_z):
        """Take a single inference step"""
        Wout = self.Wout.weight.detach().clone()  # shape [Np, Ng]
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

    def inference(self, inf_iters, inf_lr, v, prev_z, p):
        """Run inference on the hidden state"""
        self.set_nodes(v, prev_z, p)
        for i in range(inf_iters):
            with torch.no_grad():  # ensures self.z won't have grad when we call backward
                self.inference_step(inf_lr, v, prev_z)
            self.update_err_nodes(v, prev_z)

    def get_energy(self):
        """Returns the average (across batches) energy of the model"""
        if self.loss == "CE":
            obs_loss = F.cross_entropy(self.Wout(self.z), self.x)
        elif self.loss == "BCE":
            obs_loss = F.binary_cross_entropy_with_logits(self.Wout(self.z), self.x)
        else:
            obs_loss = torch.sum(self.err_x**2, -1).mean()
        latent_loss = torch.sum(self.err_z**2, -1).mean()
        energy = obs_loss + latent_loss
        energy += self.weight_decay * (torch.mean(self.Wr.weight**2))

        return energy, obs_loss
    