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
def args2dict(check=True):
    frame_func = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame_func)
    return {arg: values[arg] for arg, val in zip(args, values) if 'self' != arg or not check}

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

def __with_caller_name__(func):
    def wrapper(check=True):
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame.frame.f_globals['__name__']
        return func(caller_name, check)
    return wrapper

@__with_caller_name__
def auto_all(_name_, check=True):
    current_module = sys.modules[_name_]
    return [name for name, obj in inspect.getmembers(current_module)
            if callable(obj) and obj.__module__ == _name_ and (not name.startswith('_') if check else not name.startswith('__'))]


__all__ = auto_all(False)
