import inspect, random, sys
import numpy
import torch

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def assert_error(Bool, error, message=None):
    if not Bool:
        if message is not None:
            raise error(message)
        else:
            raise error()
def args2dict():
    frame_func = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame_func)
    return {arg: values[arg] for arg, val in zip(args, values) if 'self' != arg}

def apply_args(parent, kargs=None):
    frame_func = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame_func)
    if 'self' in args:
        args.remove('self')
    for arg in args:
        setattr(parent, arg, values[arg])
    if kargs is not None:
        apply_kwargs(parent, kargs)

def apply_kwargs(parent, kargs):
    assert isinstance(kargs, dict)
    kargs.pop('self', None)
    for arg, value in kargs.items():
        setattr(parent, arg, value)

def _with_caller_name(func):
    def wrapper():
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame.frame.f_globals['__name__']
        return func(caller_name)
    return wrapper

@_with_caller_name
def auto_all(_name_):
    current_module = sys.modules[_name_]
    return [name for name, obj in inspect.getmembers(current_module)
            if callable(obj) and obj.__module__ == _name_ and not name.startswith('_')]


__all__ = auto_all()

if __name__ == '__main__':
    class test_class:
        def __init__(self, d=None):
            print(args2dict())
            apply_args(self, d)
            print(f"self.test: {self.test}\n"
                  f"self.val: {self.val}")
    test_class({'test':1, 'val':2})