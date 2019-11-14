import time, pytest
import torch
from DominantSparseEigenAD.CG import CG_torch

def test_fullrank():
    import numpy as np
    from scipy.stats import ortho_group
    n = 100
    diagonal = 1. + 10. * np.random.rand(n)
    U = ortho_group.rvs(n)
    """
        A is randomly generated as a real, symmetric, positive definite matrix
    of size n*n.
    """
    A = U.dot(np.diag(diagonal)).dot(U.T)       
    A = torch.from_numpy(A).to(torch.float64)
    print("\n----- test_fullrank -----")
    print("----- Dimension of matrix A: %d -----" % n)

    b = torch.randn(n, dtype=torch.float64)
    initialx = torch.randn(n, dtype=torch.float64)

    start = time.time()
    x = CG_torch(A, b, initialx)
    end = time.time()
    print("CG_torch time: ", end - start)
    assert torch.allclose(A.matmul(x), b)

def test_lowrank():
    n = 300
    A = torch.randn(n, n, dtype=torch.float64)
    A = A + A.T
    eigvalues, eigvectors = torch.symeig(A, eigenvectors=True)
    alpha = eigvalues[0]
    x = eigvectors[:, 0]
    Aprime = A - alpha * torch.eye(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)
    b = b - torch.matmul(x, b) * x
    initialx = torch.randn(n, dtype=torch.float64)
    initialx = initialx - torch.matmul(x, initialx) * x
    result = CG_torch(Aprime, b, initialx)
    assert torch.allclose(torch.matmul(Aprime, result) - b, 
                          torch.zeros(n, dtype=torch.float64), 
                          atol=1e-06)
    assert torch.allclose(torch.matmul(result, x)[None], 
                          torch.zeros(1, dtype=torch.float64), 
                          atol=1e-06)

@pytest.mark.skipif(not torch.cuda.is_available(), 
        reason="No GPU support in online test envionment")
def test_fullrank_gpu():
    import numpy as np
    from scipy.stats import ortho_group
    n = 100
    diagonal = 1. + 10. * np.random.rand(n)
    U = ortho_group.rvs(n)
    """
        A is randomly generated as a real, symmetric, positive definite matrix
    of size n*n.
    """
    A = U.dot(np.diag(diagonal)).dot(U.T)       
    cuda = torch.device("cuda")
    dtype = torch.float64
    A = torch.from_numpy(A).to(cuda, dtype=dtype)

    b = torch.randn(n, device=cuda, dtype=dtype)
    initialx = torch.randn(n, device=cuda, dtype=dtype)
    x = CG_torch(A, b, initialx)
    groundtruth = torch.inverse(A).matmul(b)
    assert torch.allclose(x, groundtruth)

@pytest.mark.skipif(not torch.cuda.is_available(), 
        reason="No GPU support in online test envionment")
def test_lowrank_gpu():
    n = 300
    cuda = torch.device("cuda")
    dtype = torch.float64
    A = torch.randn(n, n, device=cuda, dtype=dtype)
    A = A + A.T
    eigvalues, eigvectors = torch.symeig(A, eigenvectors=True)
    alpha = eigvalues[0]
    x = eigvectors[:, 0]
    Aprime = A - alpha * torch.eye(n, device=cuda, dtype=dtype)
    b = torch.randn(n, device=cuda, dtype=dtype)
    b = b - torch.matmul(x, b) * x
    initialx = torch.randn(n, device=cuda, dtype=dtype)
    initialx = initialx - torch.matmul(x, initialx) * x
    result = CG_torch(Aprime, b, initialx)
    assert torch.allclose(torch.matmul(Aprime, result) - b, 
                          torch.zeros(n, device=cuda, dtype=dtype), 
                          atol=1e-06)
    assert torch.allclose(torch.matmul(result, x)[None], 
                          torch.zeros(1, device=cuda, dtype=dtype), 
                          atol=1e-06)
