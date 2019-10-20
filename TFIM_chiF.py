import numpy as np
import torch
from TFIM import TFIM
import Lanczos_torch
import CG_torch

N = 10
model = TFIM(N)
Npoints = 100
gs = np.linspace(0.3, 1.8, num=Npoints)
chiFs_perturbation = np.empty(Npoints)
chiFs_AD = np.empty(Npoints)
chiFs_sparse_AD = np.empty(Npoints)
for i in range(Npoints):
    model.g = torch.Tensor([gs[i]]).to(torch.float64)
    model.g.requires_grad_(True)

    model.setHmatrix()
    Es, psis = torch.symeig(model.Hmatrix, eigenvectors=True)
    psi0 = psis[:, 0]
    """
    dpsi0 = torch.empty(model.dim, dtype=torch.float64)
    I = torch.eye(model.dim, dtype=torch.float64)
    for idx in range(model.dim):
        dpsi0[idx], = torch.autograd.grad(psi0[idx], model.g, retain_graph=True)
    chiF_geometric = torch.matmul(dpsi0, dpsi0).item()
    """

    model.setpHpg()
    numerators = psi0.matmul(model.pHpgmatrix).matmul(psis)[1:] ** 2
    denominators = (Es[0] - Es[1:]) ** 2
    chiFs_perturbation[i] = (numerators / denominators).sum().item()

    dominant_symeig = Lanczos_torch.DominantSymeig.apply
    k = 300
    E0_AD, psi0_AD = dominant_symeig(model.Hmatrix, k)
    logF = torch.log(psi0_AD.detach().matmul(psi0_AD))
    dlogF, = torch.autograd.grad(logF, model.g, create_graph=True)
    d2logF, = torch.autograd.grad(dlogF, model.g)
    chiFs_AD[i] = -d2logF.item()
    #print("g: ", gs[i], "psi0 * dpsi0: ", torch.matmul(psi0, dpsi0), 

    CG_torch.set_CGsubspace_sparse(model.H, model.Hadjoint_to_gadjoint)
    Lanczos_torch.set_DominantSparseSymeig(model.H, model.Hadjoint_to_gadjoint)
    dominant_sparse_symeig = Lanczos_torch.DominantSparseSymeig.apply
    E0_sparse_AD, psi0_sparse_AD = dominant_sparse_symeig(model.g, k, model.dim)
    logF_sparse = torch.log(psi0_sparse_AD.detach().matmul(psi0_sparse_AD))
    dlogF_sparse, = torch.autograd.grad(logF_sparse, model.g, create_graph=True)
    d2logF_sparse, = torch.autograd.grad(dlogF_sparse, model.g)
    chiFs_sparse_AD[i] = -d2logF_sparse.item()

    print("g: ", gs[i], 
            #"chiF_geometric: ", chiF_geometric, 
            "chiF_perturbation: ", chiFs_perturbation[i], 
            "chiF_AD: ", chiFs_AD[i], 
            "chiF_sparse_AD: ", chiFs_sparse_AD[i])
import matplotlib.pyplot as plt
plt.plot(gs, chiFs_perturbation, label="perturbation")
plt.plot(gs, chiFs_AD, label="AD: normal representation")
plt.plot(gs, chiFs_sparse_AD, label="AD: sparse representation")
plt.legend()
plt.xlabel("$g$")
plt.ylabel("$\\chi_F$")
plt.title("Fidelity susceptibility of 1D TFIM\n" \
        "$H = - \\sum_{i=0}^{N-1} (g\\sigma_i^x + \\sigma_i^z \\sigma_{i+1}^z)$\n" \
        "$N=%d$" % model.N)
plt.show()
