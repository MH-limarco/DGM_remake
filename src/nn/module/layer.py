from torch import nn
from torch_geometric import nn as pyGnn

from src.nn.module import *
from src.utils.utils import *

class Linear(Module):
    def __init__(self, in_features, out_features, act=None, drop_out=0.0):
        super().__init__()
        self.conv = nn.Linear(in_features, out_features)
        self.act = activation(act) if act not in [None, True] else self.default_act
        self.drop_out = drop_out
        self.dropout = nn.Identity() if drop_out == 0 else nn.Dropout(drop_out)

    def forward(self, x, edge_index=None):
        return self.dropout(self.act(self.conv(x)))

    def __repr__(self) -> str:
        return f'{str(self.conv).replace(")", "")}, ' \
               f'activation={self.act}, ' \
               f'Dropout={self.drop_out}, '\
               f'bias={self.conv.bias is not None if hasattr(self.conv, "bias") else None})'

class pyGConv(Module):
    def __init__(self, in_features, out_features, conv, act=None, drop_out=0.0, **kwargs):
        super().__init__()
        self.conv = getattr(pyGnn, conv)(in_features, out_features)
        self.act = activation(act) if act not in [None, True] else self.default_act

        self.drop_out = drop_out
        self.dropout = nn.Identity() if drop_out == 0 else nn.Dropout(drop_out)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        return self.dropout(self.act(self.conv(x, edge_index)))

    def __repr__(self) -> str:
        return f'{str(self.conv).replace(")", "")}, ' \
               f'activation={self.act}, ' \
               f'Dropout={self.drop_out}, '\
               f'bias={self.conv.bias is not None if hasattr(self.conv, "bias") else None})'

class GCNConv(pyGConv):
    def __init__(self, in_features, out_features, act=None, drop_out=0.0):
        super(GCNConv, self).__init__(conv="GCNConv", **args2dict())

class GATConv(pyGConv):
    def __init__(self, in_features, out_features, act=None, drop_out=0.0):
        super(GATConv, self).__init__(conv="GATConv", **args2dict())

class GINConv(pyGConv):
    def __init__(self, in_features, out_features, act=None, drop_out=0.0):
        super(GINConv, self).__init__(conv="GINConv", **args2dict())

    def forward(self, x, A=None):
        raise NotImplementedError()

class GatedGraphConv(pyGConv):
    def __init__(self, in_features, out_features, num_layers , act=None, drop_out=0.0):
        super(GatedGraphConv, self).__init__(conv="GCNConv", **args2dict())
        self.conv = getattr(pyGnn, "GatedGraphConv")(out_features, num_layers)
        self.reset_parameters()

class APPNP(pyGConv):
    def __init__(self, in_features, K, alpha, act=None, drop_out=0.0, out_features=1):
        super(APPNP, self).__init__(conv="GCNConv", **args2dict())
        self.conv = getattr(pyGnn, "APPNP")(K, alpha)
        self.reset_parameters()




__all__ = auto_all()

if __name__ == "__main__":
    import torch
    x = torch.rand((64, 10))
    edge_index = torch.randint(64, size=(2, 20))
    _test_ = GCNConv(10, 1000, act=False, drop_out=0.3)
    _test_ = APPNP(10, 10, 0.5, act=False, drop_out=0.3)
    _test_ = GatedGraphConv(10, 1000, 8, act=False, drop_out=0.3)
    print(pyGnn.summary(_test_, x, edge_index, max_depth=3))
    print(_test_)

