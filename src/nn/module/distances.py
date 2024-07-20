from torch import nn

from src.nn.module import Module
from src.nn.module.layer import *
from src.utils.utils import *

class _distances_(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self._reset_parameters(self)
    def forward(self, x, edge_index=None):
        raise NotImplementedError()


class euclidean(_distances_):
    def __init__(self, dim=-1):
        super(euclidean, self).__init__(dim)
    def forward(self, x, edge_index=None):
        dist = torch.cdist(x, x) ** 2
        return dist, x

class poincare(_distances_):
    def __init__(self, dim=-1):
        super(poincare, self).__init__(dim)
    def forward(self, x, edge_index=None):
        x_norm = (x ** 2).sum(self.dim, keepdim=True)
        x_norm = (x_norm.sqrt() - 1).relu() + 1
        x = x / (x_norm * (1 + 1e-2))
        x_norm = (x ** 2).sum(self.dim, keepdim=True)

        pq = torch.cdist(x, x) ** 2
        dist = torch.arccosh(1e-6 + 1 + 2 * pq / ((1 - x_norm) * (1 - x_norm.transpose(-1, -2)))) ** 2
        return dist, x

def euclidean_distances(x, dim=-1):
    dist = torch.cdist(x, x) ** 2
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

__all__ = auto_all()
