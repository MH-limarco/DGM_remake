from collections.abc import Iterable
from torch import nn
from torch_geometric import nn as pyGnn
import torch

from src.nn.module import *
from src.nn.module.layer import *
from src.utils.utils import *


class MLP(Module):
    def __init__(self, in_features, out_features, act=None, drop_out=0.0, output_act=False, reset_parameters=True):
        super(MLP, self).__init__()
        _features_ = [in_features] + (out_features if isinstance(out_features, Iterable) else [out_features])
        self.output_act = output_act
        self.fc = nn.ModuleList()
        for _i, _o in zip(_features_[:-1], _features_[1:]):
            self.fc.append(Linear(in_features=_i, out_features=_o, act=act, drop_out=drop_out))
        self._last_layer_()

        if reset_parameters:
            self._reset_parameters(self)

    def forward(self, x, edge_index=None):
        for conv in self.fc:
            x = conv(x, edge_index)
        return x

    def _last_layer_(self):
        self.fc[-1].dropout = nn.Identity()
        self.fc[-1].drop_out = 0.0
        if not self.output_act:
            self.fc[-1].act = activation(False)

class JumpingKnowledge(pyGnn.JumpingKnowledge):
    def __init__(self, mode, channels, num_layers):
        super().__init__(**args2dict())

    def forward(self, *xs):
        return super().forward(xs)

__all__ = auto_all()

if __name__ == "__main__":
    import torch
    from torch_geometric import nn as pyGnn
    from src.nn.module.layer import *
    x = torch.rand((64, 10))
    edge_index = torch.randint(64, size=(2, 20))

    _test_0 = GCNConv(10, 1000, act=False, drop_out=0.3)
    _test_1 = MLP(1000, [20, 30, 40, 10], act=None, drop_out=0.3)

