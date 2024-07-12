from torch import nn
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import GCN

from src.utils.utils import *

class base_Module(nn.Module):
    def __init__(self, *args, **kwargs):
        super(base_Module, self).__init__()
        self.i = 0

    def _reset_parameters(self, module):
        for name, submodule in module.named_children():
            if hasattr(submodule, 'reset_parameters'):
                submodule.reset_parameters()
            self._reset_parameters(submodule)

        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    def forward(self, data):
        raise NotImplementedError()





__all__ = auto_all()
