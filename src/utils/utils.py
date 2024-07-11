import random, torch, numpy

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def assert_error(Bool, error, message=None):
    if not Bool:
        if message is not None:
            raise error(message)
        else:
            raise error()