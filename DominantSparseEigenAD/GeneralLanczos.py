import torch
import numpy as np
from scipy.sparse import linalg as sparselinalg

class DominantEig(torch.autograd.Function):
    """
        Function primitive of dominant eigensolver, where the matrix is real
    and only assumed to be diagonalizable. In addition, since Pytorch doesn't
    support complex numbers, the desired eigenvalue of the matrix (hence in turn
    the corresponding left/right eigenvectors) should be real.

    input: A -- the real matrix A.
           k -- number of Lanczos vectors requested.(doesn't need gradient)
    output: eigval -- the largest-amplitude eigenvalue of A.
            lefteigvector -- corresponding (non-degenerate) left eigenvector l.
            righteigvector -- corresponding (non-degenerate) right eigenvector r.
                Conventionally, the left and right eigenvector obey the orthogonal
            relation l^T r = 1. In addition, the normalization r^T r = 1 is chosen,
            but this shouldn't have any effect on gauge invariant computation process.
    """
    @staticmethod
    def forward(ctx, A, k):
        A_numpy = A.detach().numpy()
        righteigval, righteigvector = sparselinalg.eigs(A_numpy, k=1, which="LM", ncv=k)
        lefteigval, lefteigvector = sparselinalg.eigs(A_numpy.T, k=1, which="LM", ncv=k)
        assert np.allclose(righteigval.imag, 0.0), \
                "The desired eigenvalue of the matrix must be real"
        eigval = righteigval.real
        righteigvector = righteigvector[:, 0].real
        lefteigvector = lefteigvector[:, 0].real
        lefteigvector /= np.dot(lefteigvector, righteigvector)
        ctx.A = A_numpy
        ctx.eigval, ctx.lefteigvector, ctx.righteigvector = \
                eigval, lefteigvector, righteigvector
        return torch.from_numpy(eigval), \
               torch.from_numpy(lefteigvector), \
               torch.from_numpy(righteigvector)

    @staticmethod
    def backward(ctx, grad_eigval, grad_lefteigvector, grad_righteigvector):
        A = ctx.A
        eigval, lefteigvector, righteigvector = \
                ctx.eigval, ctx.lefteigvector, ctx.righteigvector
        grad_lefteigvector = grad_lefteigvector.numpy()
        grad_righteigvector = grad_righteigvector.numpy()

        Aprime = A - eigval * np.eye(A.shape[0])
        b = grad_lefteigvector - righteigvector * np.dot(lefteigvector, grad_lefteigvector)
        lambdal0, _ = sparselinalg.gmres(Aprime, b, tol=1e-12, atol=1e-12)
        Aprime = A.T - eigval * np.eye(A.shape[0])
        b = grad_righteigvector - lefteigvector * np.dot(righteigvector, grad_righteigvector)
        lambdar0, _ = sparselinalg.gmres(Aprime, b, tol=1e-12, atol=1e-12)
        grad_A = grad_eigval.numpy() * lefteigvector[:, None] * righteigvector \
                    - lefteigvector[:, None] * lambdal0 \
                    - lambdar0[:, None] * righteigvector
        grad_A, grad_k = torch.from_numpy(grad_A), None
        return grad_A, grad_k
