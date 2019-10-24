########
#    A simple, artificial 2-dimensional low-rank linear system.
#
#    Used for correctness-checking and demonstration of the
#CG_subspace torch.autograd.Function primitive.
########

import sys
sys.path.append("..")
from CG_torch import CGSubspace
import torch

# Constructing a simple computation graph
A0 = torch.randn(2, 2, dtype=torch.float64)
A0 = A0 + A0.T
b0 = torch.randn(2, dtype=torch.float64)
flip = torch.Tensor([[0, -1], [1, 0]]).to(torch.float64)
while(True):
    alpha0 = torch.randn(2, dtype=torch.float64)
    alpha0 = alpha0 / torch.norm(alpha0)
    alpha = flip.matmul(alpha0)
    if alpha.matmul(A0).matmul(alpha) > 0:
        break
alpha0.requires_grad_(True)
alpha = flip.matmul(alpha0)
z = torch.randn(2, dtype=torch.float64)

A = alpha.matmul(A0).matmul(alpha) * (alpha[:, None] * alpha)
b = torch.matmul(alpha, b0) * alpha

# Forward and backward of CG
CG = CGSubspace.apply
x = CG(A, b, alpha0)
dresult, = torch.autograd.grad(torch.matmul(x, z), alpha0)

# Analytic results
x_analytic = torch.matmul(alpha, b0) * alpha \
             / torch.matmul(alpha, alpha) \
             / alpha.matmul(A0).matmul(alpha)
dresult_analytic = torch.matmul(x_analytic, z) * \
                (b0 / torch.matmul(alpha, b0) + z / torch.matmul(alpha, z) \
                - 2 * alpha / torch.matmul(alpha, alpha) \
                - 2 * torch.matmul(A0, alpha) / alpha.matmul(A0).matmul(alpha))
dresult_analytic = flip.T.matmul(dresult_analytic)
print("x\t\tanalytic: ", x_analytic, "\tCG_forward: ", x)
print("dresult\t\tanalytic: ", dresult_analytic, "\tCG_backward: ", dresult)
