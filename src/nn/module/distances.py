import torch

from src.nn.module.layer import base_Module
from src.utils.utils import *

def euclidean_distances(x, dim=-1):
    dist = torch.cdist(x ,x ) ** 2
    return dist, x

# #Poincarè disk distance r=1 (Hyperbolic)
def poincare_distances(x, dim=-1):
    x_norm = (x ** 2).sum(dim ,keepdim=True)
    x_norm = (x_norm.sqrt( ) -1).relu() + 1
    x = x/ (x_norm * (1 + 1e-2))
    x_norm = (x ** 2).sum(dim, keepdim=True)

    pq = torch.cdist(x, x) ** 2
    dist = torch.arccosh(1e-6 + 1 + 2 * pq / ((1 - x_norm) * (1 - x_norm.transpose(-1, -2)))) ** 2
    return dist, x

class _learnable_distances(base_Module):
    def __init__(self):
        super(_learnable_distances, self).__init__()
        ##TODO

        self._reset_parameters(self)

    def forward(self, x, A=None):
        pass

__all__ = auto_all()
