########
#    A simple correctness-check of the DominantSymeig torch.autograd.Function
#primitive.
########

import numpy as np
import torch

"""
xmin, xmax, N = -1., 1., 300
h = (xmax - xmin) / N
xmesh = np.linspace(xmin, xmax, num=N, endpoint=False)
target = np.zeros(N)
idx = (np.abs(xmesh) < 0.5)
target[idx] = 1. - np.abs(xmesh[idx])
target /= np.linalg.norm(target)
xmesh = torch.from_numpy(xmesh)
target = torch.from_numpy(target)

K = -0.5/h**2 * (torch.diag(-2 * torch.ones(N, dtype=xmesh.dtype))
                + torch.diag(torch.ones(N - 1, dtype=xmesh.dtype), diagonal=1)
                + torch.diag(torch.ones(N - 1, dtype=xmesh.dtype), diagonal=-1))
"""
N = 300
K = torch.randn(N, N, dtype=torch.float64)
K = K + K.T
target = torch.randn(N, dtype=torch.float64)
#potential = 0.5 * xmesh**2
#potential = torch.randn(N, dtype=xmesh.dtype)
potential = torch.randn(N, dtype=torch.float64)
potential.requires_grad_(True)
V = torch.diag(potential)
H = K + V

# Backward through torch.symeig implemented by Pytorch.
Es, psis = torch.symeig(H, eigenvectors=True)
psi0_torch = psis[:, 0]
loss_torch = 1. - torch.matmul(psi0_torch, target)
grad_potential_torch, = torch.autograd.grad(loss_torch, potential, retain_graph=True)
print("loss: ", loss_torch)
print("GradV from PyTorch implementation: ", \
        grad_potential_torch)

# Backward through the DominantSymeig primitive.
import sys
sys.path.append("..")
from Lanczos_torch import DominantSymeig

dominant_symeig = DominantSymeig.apply
k = 100
_, psi0_dominant = dominant_symeig(H, k)
loss_dominant = 1. - torch.matmul(psi0_dominant, target)
grad_potential_dominant, = torch.autograd.grad(loss_dominant, potential)
print("loss: ", loss_dominant)
print("GradV from DominantSymeig primitive: ", \
        grad_potential_dominant)

print("The difference: ", grad_potential_torch.abs() - grad_potential_dominant.abs())
