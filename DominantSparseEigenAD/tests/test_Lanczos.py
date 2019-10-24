import torch
from ..Lanczos import symeigLanczos, DominantSymeig

def test_normal():
    import time
    n = 1000
    A = 0.1 * torch.rand(n, n, dtype=torch.float64)
    A = A + A.T
    k = 300
    print("----- Dimension of real symmetric matrix A: %d -----" % n)
    print("Running times: ")

    start = time.time()
    eigval_min, eigvector_min, eigval_max, eigvector_max = symeigLanczos(A, k)
    end = time.time()
    print("Lanczos: ", end - start)

    start = time.time()
    eigvals, eigvectors = torch.symeig(A, eigenvectors=True)
    end = time.time()
    print("Pytorch: ", end - start)

    assert torch.allclose(eigval_min, eigvals[0])
    assert torch.allclose(eigval_max, eigvals[-1])
    assert torch.allclose(eigvector_min, eigvectors[:, 0]) or \
            torch.allclose(eigvector_min, -eigvectors[:, 0])
    assert torch.allclose(eigvector_max, eigvectors[:, -1]) or \
            torch.allclose(eigvector_max, -eigvectors[:, -1])

def test_sparse():
    n = 1000
    A = 0.1 * torch.rand(n, n, dtype=torch.float64)
    A = A + A.T
    dim = A.shape[0]
    Amap = lambda v: torch.matmul(A, v)
    k = 300
    eigval_min, eigvector_min, eigval_max, eigvector_max = symeigLanczos(Amap, k, sparse=True, dim=dim)
    eigvals, eigvectors = torch.symeig(A, eigenvectors=True)
    assert torch.allclose(eigval_min, eigvals[0])
    assert torch.allclose(eigval_max, eigvals[-1])
    assert torch.allclose(eigvector_min, eigvectors[:, 0]) or \
            torch.allclose(eigvector_min, -eigvectors[:, 0])
    assert torch.allclose(eigvector_max, eigvectors[:, -1]) or \
            torch.allclose(eigvector_max, -eigvectors[:, -1])

def test_DominantSymeig():
    N = 300
    K = torch.randn(N, N, dtype=torch.float64)
    K = K + K.T
    target = torch.randn(N, dtype=torch.float64)
    potential = torch.randn(N, dtype=torch.float64)
    potential.requires_grad_(True)
    V = torch.diag(potential)
    H = K + V

    # Backward through torch.symeig implemented by Pytorch.
    Es, psis = torch.symeig(H, eigenvectors=True)
    psi0_torch = psis[:, 0]
    loss_torch = 1. - torch.matmul(psi0_torch, target)
    grad_potential_torch, = torch.autograd.grad(loss_torch, potential, retain_graph=True)
    #print("Pytorch: ")
    #print("loss: ", loss_torch, "grad_V: ", grad_potential_torch)

    # Backward through the DominantSymeig primitive.
    dominant_symeig = DominantSymeig.apply
    k = 100
    _, psi0_dominant = dominant_symeig(H, k)
    loss_dominant = 1. - torch.matmul(psi0_dominant, target)
    grad_potential_dominant, = torch.autograd.grad(loss_dominant, potential)
    #print("DominantSymeig: ")
    #print("loss: ", loss_dominant, "grad_V: ", grad_potential_dominant)

    assert torch.allclose(loss_dominant, loss_torch) or \
           torch.allclose(loss_dominant, 2. - loss_torch)
    assert torch.allclose(grad_potential_dominant, grad_potential_torch) or \
           torch.allclose(grad_potential_dominant, -grad_potential_torch) 
