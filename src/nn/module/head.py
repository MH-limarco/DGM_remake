from collections.abc import Iterable
from torch import nn
from torch_geometric import nn as pyGnn
import torch

from src.nn.module import Module
from src.nn.module.block import *
from src.utils.utils import *

class Classify(Module):
    def __init__(self, in_features, out_features):
        super(Classify, self).__init__()
        self.mlp = MLP(in_features, out_features)

    def forward(self, x, edge_index):
        return self.mlp(x)


__all__ = auto_all()
