import torch
from DominantSparseEigenAD.CG import CG_torch, CGSubspace

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

    b = torch.randn(n, dtype=torch.float64)
    initialx = torch.randn(n, dtype=torch.float64)
    x = CG_torch(A, b, initialx)
    groundtruth = torch.inverse(A).matmul(b)
    assert torch.allclose(x, groundtruth)

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
