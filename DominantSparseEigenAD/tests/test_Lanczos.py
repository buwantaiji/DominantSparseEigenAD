import time
import torch
from DominantSparseEigenAD.Lanczos import symeigLanczos, DominantSymeig

def test_normal():
    n = 1000
    k = 300
    print("\n----- test_normal -----")
    print("----- Dimension of real symmetric matrix A: %d -----" % n)
    print("Running times: ")
    for i in range(1):
        A = 0.1 * torch.rand(n, n, dtype=torch.float64)
        A = A + A.T

        start = time.time()
        eigval_min, eigvector_min, eigval_max, eigvector_max = symeigLanczos(A, k)
        end = time.time()
        print("Lanczos: ", end - start, end="    ")

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

def test_normal_gpu():
    n = 1000
    k = 300
    cuda = torch.device("cuda")
    print("\n----- test_normal_gpu -----")
    print("----- Dimension of real symmetric matrix A: %d -----" % n)
    print("Running times: ")
    for i in range(1):
        A = 0.1 * torch.rand(n, n, dtype=torch.float64, device=cuda)
        A = A + A.T

        start = time.time()
        eigval_min, eigvector_min, eigval_max, eigvector_max = symeigLanczos(A, k, 
                device=cuda)
        end = time.time()
        print("Lanczos: ", end - start, end="    ")

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
    k = 300
    A = 0.1 * torch.rand(n, n, dtype=torch.float64)
    A = A + A.T
    dim = A.shape[0]
    Amap = lambda v: torch.matmul(A, v)
    eigval_min, eigvector_min, eigval_max, eigvector_max = symeigLanczos(Amap, k, sparse=True, dim=dim)
    eigvals, eigvectors = torch.symeig(A, eigenvectors=True)
    assert torch.allclose(eigval_min, eigvals[0])
    assert torch.allclose(eigval_max, eigvals[-1])
    assert torch.allclose(eigvector_min, eigvectors[:, 0]) or \
            torch.allclose(eigvector_min, -eigvectors[:, 0])
    assert torch.allclose(eigvector_max, eigvectors[:, -1]) or \
            torch.allclose(eigvector_max, -eigvectors[:, -1])

def test_sparse_gpu():
    n = 1000
    k = 300
    cuda = torch.device("cuda")
    A = 0.1 * torch.rand(n, n, dtype=torch.float64, device=cuda)
    A = A + A.T
    dim = A.shape[0]
    Amap = lambda v: torch.matmul(A, v)
    eigval_min, eigvector_min, eigval_max, eigvector_max = symeigLanczos(Amap, k, 
            device=cuda, sparse=True, dim=dim)
    eigvals, eigvectors = torch.symeig(A, eigenvectors=True)
    assert torch.allclose(eigval_min, eigvals[0])
    assert torch.allclose(eigval_max, eigvals[-1])
    assert torch.allclose(eigvector_min, eigvectors[:, 0]) or \
            torch.allclose(eigvector_min, -eigvectors[:, 0])
    assert torch.allclose(eigvector_max, eigvectors[:, -1]) or \
            torch.allclose(eigvector_max, -eigvectors[:, -1])

def test_normal_tridiagonal():
    import numpy as np
    xmin, xmax, N = -1., 1., 1000
    xmesh = np.linspace(xmin, xmax, num=N, endpoint=False)
    xmesh = torch.from_numpy(xmesh).to(torch.float64)
    h = (xmax - xmin) / N
    K = -0.5/h**2 * (torch.diag(-2 * torch.ones(N, dtype=xmesh.dtype))
                    + torch.diag(torch.ones(N - 1, dtype=xmesh.dtype), diagonal=1)
                    + torch.diag(torch.ones(N - 1, dtype=xmesh.dtype), diagonal=-1))
    potential = 0.5 * xmesh**2
    V = torch.diag(potential)
    Hmatrix = K + V
    k = 1000

    start = time.time()
    E0, psi0 = symeigLanczos(Hmatrix, k, extreme="min")
    end = time.time()
    print("\n----- test_normal_tridiagonal -----")
    print("Lanczos: ", end - start)

    start = time.time()
    Es, psis = torch.symeig(Hmatrix, eigenvectors=True)
    end = time.time()
    print("Pytorch: ", end - start)
    
    assert torch.allclose(E0, Es[0])
    assert torch.allclose(psi0, psis[:, 0]) or \
           torch.allclose(psi0, -psis[:, 0])

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
