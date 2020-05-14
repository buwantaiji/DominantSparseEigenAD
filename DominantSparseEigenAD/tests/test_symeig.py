import torch
import time
from DominantSparseEigenAD.symeig import DominantSymeig

def test_DominantSymeig():
    N = 300
    K = torch.randn(N, N, dtype=torch.float64)
    K = K + K.T
    target = torch.randn(N, dtype=torch.float64)
    potential = torch.randn(N, dtype=torch.float64)
    potential.requires_grad_(True)
    V = torch.diag(potential)
    H = K + V
    print("\n----- test_DominantSymeig -----")
    print("----- Dimension of the Hamiltonian: %d -----" % N)

    for i in range(3):
        ##### Backward through torch.symeig implemented by Pytorch.
        start1 = time.time()
        Es, psis = torch.symeig(H, eigenvectors=True)
        psi0_torch = psis[:, 0]
        loss_torch = 1. - torch.matmul(psi0_torch, target)
        end1 = time.time()

        start2 = time.time()
        grad_potential_torch, = torch.autograd.grad(loss_torch, potential)
        end2 = time.time()
        print("Pytorch\t\tforward time: %f    backward time: %f" % (end1 - start1, end2 - start2))

        ##### Backward through the DominantSymeig primitive.
        dominant_symeig = DominantSymeig.apply
        k = 300
        start1 = time.time()
        _, psi0_dominant = dominant_symeig(H, k)
        loss_dominant = 1. - torch.matmul(psi0_dominant, target)
        end1 = time.time()

        start2 = time.time()
        grad_potential_dominant, = torch.autograd.grad(loss_dominant, potential)
        end2 = time.time()
        print("DominantSymeig\tforward time: %f    backward time: %f" % (end1 - start1, end2 - start2))

        assert torch.allclose(loss_dominant, loss_torch) or \
               torch.allclose(loss_dominant, 2. - loss_torch)
        assert torch.allclose(grad_potential_dominant, grad_potential_torch) or \
               torch.allclose(grad_potential_dominant, -grad_potential_torch) 

