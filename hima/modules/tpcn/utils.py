import torch
import torch.nn as nn


class Softmax(nn.Module):
    def forward(self, inp):
        return torch.softmax(inp, dim=-1)

    def deriv(self, inp):
        # Compute the softmax output
        soft = self.forward(inp)
        # Initialize a tensor for the derivative, with the same shape as the softmax output
        s = soft.unsqueeze(-2)  # Add a dimension for broadcasting
        identity = torch.eye(s.size(-1)).unsqueeze(0).to(inp.device)  
        # The diagonal contains s_i * (1 - s_i) and off-diagonal s_i * (-s_j)
        deriv = identity * s - s * s.transpose(-1, -2) # shape (batch_size, N, N)
        return deriv

class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

class ReLU(nn.Module):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out

class Sigmoid(nn.Module):
    def forward(self, inp):
        return torch.sigmoid(inp)

    def deriv(self, inp):
        out = self(inp)
        return out * (1 - out)
    