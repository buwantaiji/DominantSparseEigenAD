"""
    A small toy example demonstrating how the process of computing 1st
derivative can be added to the original computation graph to produce an enlarged
graph whose back-propagation yields the 2nd derivative.
"""
import torch

x = torch.randn(10, requires_grad=True)
exp = torch.exp(x)
cos = torch.cos(x)
y = exp * cos
cosbar = exp
expbar = cos
minussin = -torch.sin(x)
grad1 = cosbar * minussin
grad2 = expbar * exp
dydx = grad1 + grad2
d2ydx2 = torch.autograd.grad(dydx, x, grad_outputs=torch.ones(dydx.shape[0]))

print("y: ", y, "\ngroundtruth: ", torch.exp(x) * torch.cos(x))
print("dy/dx: ", dydx, "\ngroundtruth: ", torch.exp(x) * (torch.cos(x)- torch.sin(x)))
print("d2y/dx2: ", d2ydx2, "\ngroundtruth", -2 * torch.exp(x) * torch.sin(x))
