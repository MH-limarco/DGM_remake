from torch import nn
from src.utils.utils import auto_all, assert_error

def activation(_name_):
    if isinstance(_name_, str):
        try:
            return getattr(nn, _name_)(inplace=True)
        except TypeError:
            return getattr(nn, _name_)()
        except AttributeError:
            assert_error(False, AttributeError, 1235)
    else:
        return nn.Identity()

class Identity(nn.Module):
    def __init__(self, param_idx=None):
        super(Identity, self).__init__()
        self.param_idx = param_idx

    def forward(self, *params):
        if self.param_idx is not None:
            return params[self.param_idx]
        return params

class Module(nn.Module):
    default_act = activation("SiLU")
    def __init__(self):
        super(Module, self).__init__()

    def _reset_parameters(self, module):
        for name, submodule in module.named_children():
            if hasattr(submodule, 'reset_parameters'):
                submodule.reset_parameters()
            self._reset_parameters(submodule)

        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    def forward(self, x, A=None):
        raise NotImplementedError()


__all__ = auto_all(False)

if __name__ == "__main__":
    print(activation("ReLU"))
