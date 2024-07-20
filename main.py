import torch
from pykeops.torch import Genred, LazyTensor

# Notice that the parameter gamma is a dim-1 vector, *not* a scalar:
gamma  = torch.tensor([.5])
# Generate the data as pytorch tensors. If you intend to compute gradients, don't forget the `requires_grad` flag!
x = torch.randn(1000,3)
y = torch.randn(2000,3)
b = torch.randn(2000,2)

gaussian_conv = Genred(
    'Zero(-G)',  # F(g,x,y,b) = exp( -g*|x-y|^2 ) * b
    'A = Vi(2)',          # Output indexed by "i",        of dim 2
    'G = Pm(1)',          # First arg  is a parameter,    of dim 1
    )         # Fourth arg is indexed by "j", of dim 2

formula = 'Zero(-G)'
aliases = [
           'G = Pm(1)']

print(x)
x = LazyTensor(x[None, :, :])

print(x.xlogx().argmax())

