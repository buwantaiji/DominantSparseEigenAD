import torch
import numpy as np
from scipy.sparse import linalg as sparselinalg

class DominantEig(torch.autograd.Function):
    """
        Function primitive of dominant eigensolver, where the matrix is real
    and only assumed to be diagonalizable. In addition, since Pytorch doesn't
    support complex numbers, the desired eigenvalue of the matrix (hence in turn
    the corresponding left/right eigenvectors) should be real.
        In this primitive, the matrix is represented in normal form as a torch.Tensor.

    input: A -- the real matrix A.
           k -- number of Lanczos vectors requested.(doesn't need gradient)
           which -- a string indicating which eigenvalus and corresponding eigenvectors
                    to find. It can take one of the following values:
                    "LM"(default): largest magnitude;    "SM": smallest magnitude;
                    "LR": largest real part;    "SR": smallest real part.
    output: eigval -- the desired eigenvalue of A, as specified by the argument "which".
            lefteigvector -- corresponding (non-degenerate) left eigenvector l.
            righteigvector -- corresponding (non-degenerate) right eigenvector r.
                Conventionally, the left and right eigenvector obey the orthogonal
            relation l^T r = 1. In addition, the normalization r^T r = 1 is chosen,
            but this shouldn't have any effect on gauge invariant computation process.
    """
    @staticmethod
    def forward(ctx, A, k, which="LM"):
        A_numpy = A.detach().numpy()
        righteigval, righteigvector = sparselinalg.eigs(A_numpy, k=1, which=which, ncv=k)
        lefteigval, lefteigvector = sparselinalg.eigs(A_numpy.T, k=1, which=which, ncv=k)
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
        grad_A, grad_which, grad_k = torch.from_numpy(grad_A), None, None
        return grad_A, grad_which, grad_k

def setDominantSparseEig(A, AT, Aadjoint_to_gadjoint):
    """
        Function primitive of dominant eigensolver, where the matrix is real
    and only assumed to be diagonalizable. In addition, since Pytorch doesn't
    support complex numbers, the desired eigenvalue of the matrix (hence in turn
    the corresponding left/right eigenvectors) should be real.

        In this primitive, the matrix is "sparse" and represented as a
    scipy.sparse.linalg.LinearOperator.

        As a workaround of the fact that Pytorch doesn't support taking gradient 
    of objects of type other than torch.Tensor, the computation graph of this primitive 
    is slightly wrapped compared to DominantEig, the version of the case
    of normal torch.Tensor representation, and is schematically shown below.
        ---------
        |     --|--> eigval
        |    /  |
    g --|-->A---|--> lefteigvector
        |    \  |
        |     --|--> righteigvector
        ---------

    input: g -- The parameter(s) of interest of the matrix A, whose gradients are requested.
                In current version, g must be a torch.Tensor of arbitrary shape.
           k -- number of Lanczos vectors requested. (doesn't need gradient)
    output: eigval -- the largest-amplitude eigenvalue of A.
            lefteigvector -- corresponding (non-degenerate) left eigenvector l.
            righteigvector -- corresponding (non-degenerate) right eigenvector r.
                Conventionally, the left and right eigenvector obey the orthogonal
            relation l^T r = 1. In addition, the normalization r^T r = 1 is chosen,
            but this shouldn't have any effect on gauge invariant computation process.
    
    USER NOTE: The mechanism of wrapping relies on user's providing three quantities:
        A, AT -- The "sparse" representation of the matrix A and A.T (i.e., 
            A's transpose) as a scipy.sparse.linalg.LinearOperator, respectively.

        Aadjoint_to_gadjoint -- A function that receives the adjoint of the matrix A
            as input, and returns the adjoint of the pamameter(s) g as output.

            The input should be of the form ((u1, v1), (u2, v2), (u3, v3)), i.e., 
        a tuple of three pairs. The us and vs are all vectors represented as
        np.ndarrays and have size (N,), where N is the dimension of the matrix A.

            The adjoint of A = u1 * v1^T + u2 * v2^T + u3 * v3^T. (i.e., sum of 
        outer products). User may do whatever he wants to get the adjoint of g 
        using these vectors. Note that the final result (i.e., the adjoint of g)
        should be returned as a torch.Tensor.
    """
    global DominantSparseEig

    @staticmethod
    def forward(ctx, g, k):
        righteigval, righteigvector = sparselinalg.eigs(A, k=1, which="LM", ncv=k)
        lefteigval, lefteigvector = sparselinalg.eigs(AT, k=1, which="LM", ncv=k)
        assert np.allclose(righteigval.imag, 0.0), \
                "The desired eigenvalue of the matrix must be real"
        eigval = righteigval.real
        righteigvector = righteigvector[:, 0].real
        lefteigvector = lefteigvector[:, 0].real
        lefteigvector /= np.dot(lefteigvector, righteigvector)
        ctx.eigval, ctx.lefteigvector, ctx.righteigvector = \
                eigval, lefteigvector, righteigvector
        return torch.from_numpy(eigval), \
               torch.from_numpy(lefteigvector), \
               torch.from_numpy(righteigvector)

    @staticmethod
    def backward(ctx, grad_eigval, grad_lefteigvector, grad_righteigvector):
        eigval, lefteigvector, righteigvector = \
                ctx.eigval, ctx.lefteigvector, ctx.righteigvector
        grad_eigval, grad_lefteigvector, grad_righteigvector = \
                grad_eigval.numpy(), grad_lefteigvector.numpy(), grad_righteigvector.numpy()

        matvec = lambda v: A.matvec(v) - eigval * v
        Aprime = sparselinalg.LinearOperator(A.shape, matvec=matvec)
        b = grad_lefteigvector - righteigvector * np.dot(lefteigvector, grad_lefteigvector)
        lambdal0, _ = sparselinalg.gmres(Aprime, b, tol=1e-12, atol=1e-12)
        matvec = lambda v: AT.matvec(v) - eigval * v
        Aprime = sparselinalg.LinearOperator(AT.shape, matvec=matvec)
        b = grad_righteigvector - lefteigvector * np.dot(righteigvector, grad_righteigvector)
        lambdar0, _ = sparselinalg.gmres(Aprime, b, tol=1e-12, atol=1e-12)
        grad_A = (grad_eigval * lefteigvector, righteigvector), \
                    (-lefteigvector, lambdal0), \
                    (-lambdar0, righteigvector)
        grad_g, grad_k = Aadjoint_to_gadjoint(grad_A), None
        return grad_g, grad_k

    DominantSparseEig = type("DominantSparseEig", (torch.autograd.Function, ), 
            {"forward": forward, "backward": backward})
