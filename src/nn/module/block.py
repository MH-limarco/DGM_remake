from torch import nn
import torch

from src.nn.module.layer import *
from collections.abc import Iterable

class MLP(base_Module):
    def __init__(self, in_features, out_features, reset_parameters=True):
        super(MLP, self).__init__()
        _features_ = [in_features] + (out_features if isinstance(out_features, Iterable) else [out_features])
        self.fc = nn.ModuleList()
        for _i, _o in zip(_features_[:-1], _features_[1:]):
            self.fc.append(nn.Linear(in_features=_i, out_features=_o))

        if reset_parameters:
            self._reset_parameters(self)

    def forward(self, x, A=None):
        for conv in self.fc:
            x = conv(x)
        return x


class Identity(nn.Module):
    def __init__(self, param_idx=None):
        super(Identity, self).__init__()
        self.param_idx = param_idx

    def forward(self, *params):
        if self.param_idx is not None:
            return params[self.param_idx]

        return params



if __name__ == "__main__":
    MLP(10, [20,30,40,10])