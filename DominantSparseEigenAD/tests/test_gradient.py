import torch
import numpy as np
from DominantSparseEigenAD.eig import DominantEig

def test_eigs_gradient():
    D = 5
    d = 2
    A = np.random.randn(d, D, D)
    Gong = np.einsum("kij,kmn->imjn", A, A.conj()).reshape(D**2, D**2)
    Gong = torch.from_numpy(Gong)
    Gong.requires_grad_()
    k = 25

    a = torch.randn(1, dtype=torch.float64)
    Arandom = torch.randn(D**2, D**2, dtype=torch.float64)
    dominant_eig = DominantEig.apply
    def func(A, k):
        eigval, lefteigvector, righteigvector = dominant_eig(A, k)
        result = a * eigval + lefteigvector.matmul(Arandom).matmul(righteigvector)
        return result

    torch.autograd.gradcheck(func, (Gong, k))
