from torch import nn
from src.utils.utils import auto_all, assert_error

def activation(_name_):
    try:
        return getattr(nn, _name_)(inplace=True)
    except TypeError:
        return getattr(nn, _name_)()
    except AttributeError:
        assert_error(False, AttributeError, 1235)


__all__ = auto_all()

if __name__ == "__main__":
    print(activation("ReLU"))