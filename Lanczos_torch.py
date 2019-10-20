import torch

def Lanczos(A, k, *, sparse=False, dim=None):
    """
        Lanczos iteration algorithm on a real symmetric matrix using Pytorch.

    Input: `A` is a real symmetrix matrix of size, say, n.
           `k` is the number of Lanczos vectors requested. In typical applications, k << n.
           `sparse` indicates whether a bare linear function representation of the
                matrix A is adopted. When it is `true`, the integer parameter `dim`
                must be supplied giving the actual dimension of square matrix A.
    Output: A tuple (Qk, T), where Qk = (q1 q2 ... qk) is a n*k matrix, 
            whose columns contain k orthomormal Lanczos vectors q1, q2, ..., qk.
            I.e., we have Qk^T * Qk = I_k, where I_k is the k-dimensional identity matrix.
            T is a tridiagonal matrix of size k.

        Theoretically, when input k = n, the corresponding outputs Qn and T satisfy
    Qn^T * A * Qn = T, and the eigenvalues and eigenvectors of T will be identically
    the same as that of original matrix A.
        However, for any k = 1, 2, ..., n, the Lanczos vectors q1, q2, ..., qk are
    carefully selected in this algorithm that they constitute a orthonormal basis
    of Krylov space K(A, q1, k) = span{q1, A*q1, ..., A^(k-1)*q1}. Then it can be
    shown that the eigenvalues in the extreme region(i.e., closed to largest and
    smallest eigenvalues) and corresponding eigenvectors can be accurately
    approximated by the results obtained by solving the original eigenvalue problem
    restricted in the Krylov subspace K(A, q1, k), even though its dimension k
    (the number of iterations actually performed) is FAR LESS THAN n.

        In practice, after this subroutine is called, one can diagonalize
    the tridiagonal matrix T to get accurate approximations of eigenvalues of A in
    the extreme region. The corresponding eigenvectors are obtained by multiplying
    the "eigenvector representation in the k-dimensional Krylov subspace",
    which is a k-dimensional vector, by the matrix Qk.

        In practice, the Lanczos iteration turns out to be unstable upon
    floating point arithmetic. The basic difficulty is caused by loss of orthogonality
    among the Lanczos vectors, which would mess up the actual result of eigenvectors.
    In current version, the simple but a bit expensive "full reorthogonalization"
    approach is adopted to cure this problem.
    """
    if sparse:
        n = dim
        dtype = torch.float64
        Amap = A
    else:
        n = A.shape[0]
        dtype = A.dtype
        Amap = lambda v: torch.matmul(A, v)
    Qk = torch.zeros((n, k), dtype=dtype)
    alphas = torch.zeros(k, dtype=dtype)
    betas = torch.zeros(k - 1, dtype=dtype)
    q = torch.randn(n, dtype=dtype)
    q = q / torch.norm(q)
    u = Amap(q)
    alpha = torch.matmul(q, u)
    Qk[:, 0] = q
    alphas[0] = alpha
    beta = 0
    qprime = torch.randn(n, dtype=dtype)
    for i in range(1, k):
        r = u - alpha * q - beta * qprime

        #   The simple but expensive full reorthogonalization process
        # in order to recover the orthogonality among the Lanczos vectors caused by
        # rounding error in floating point arithmetic.
        r -= torch.matmul(Qk[:, :i], torch.matmul(Qk[:, :i].T, r))

        qprime = q
        beta = torch.norm(r)
        q = r / beta
        u = Amap(q)
        alpha = torch.matmul(q, u)
        alphas[i] = alpha
        betas[i - 1] = beta
        Qk[:, i] = q
    T = torch.diag(alphas) + torch.diag(betas, diagonal=1) + torch.diag(betas, diagonal=-1)
    return Qk, T

def symeigLanczos(A, k, extreme="both", *, sparse=False, dim=None):
    """
        This function computes the extreme(minimum or maximum, or both) eigenvalues
    and corresponding eigenvectors of a real symmetric matrix A based on Lanczos algorithm.

    Input: `A` is the real symmetric matrix to be diagonalized.
           `k` is the number of Lanczos vectors requested.
           `sparse` indicates whether a bare linear function representation of the
                matrix A is adopted. When it is `true`, the integer parameter `dim`
                must be supplied giving the actual dimension of square matrix A.
           `extreme` labels which of the two extreme eigenvalues and corresponding eigenvectors are returned.
            "both" -> both min and max.     --Output--> (eigval_min, eigvector_min, eigval_max, eigvector_max)
            "min" -> min.                   --Output--> (eigval_min, eigvector_min)
            "max" -> max.                   --Output--> (eigval_max, eigvector_max)
    Output: As shown in "Input" section above. Note all the elements of
            returned tuples are torch Tensors, including the eigenvalues.
    """
    Qk, T = Lanczos(A, k, sparse=sparse, dim=dim)
    eigvalsQ, eigvectorsQ = torch.symeig(T, eigenvectors=True)
    eigvectorsQ = torch.matmul(Qk, eigvectorsQ)
    if extreme == "both":
        return eigvalsQ[0], eigvectorsQ[:, 0], eigvalsQ[-1], eigvectorsQ[:, -1]
    elif extreme == "min":
        return eigvalsQ[0], eigvectorsQ[:, 0]
    elif extreme == "max":
        return eigvalsQ[-1], eigvectorsQ[:, -1]

class DominantSymeig(torch.autograd.Function):
    """
        Function primitive of dominant real symmetric eigensolver, where the matrix
    is represented in normal form as a torch.Tensor.
    
    input: A -- the real symmetric matrix A.
           k -- number of Lanczos vectors requested.(doesn't need gradient)
    output: eigval -- the smallest eigenvalue of A.
            eigvector -- corresponding (non-degenerate) eigenvector.
    """ 
    @staticmethod
    def forward(ctx, A, k):
        eigval, eigvector = symeigLanczos(A, k, extreme="min")
        ctx.save_for_backward(A, eigval, eigvector)
        return eigval, eigvector
    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvector):
        from CG_torch import CGSubspace
        A, eigval, eigvector = ctx.saved_tensors
        Aprime = A - eigval * torch.eye(A.shape[0], dtype=A.dtype)
        CG = CGSubspace.apply
        b = grad_eigvector - torch.matmul(eigvector, grad_eigvector) * eigvector
        lambda0 = CG(Aprime, b, eigvector)
        grad_A = (grad_eigval * eigvector - lambda0)[:, None] * eigvector
        grad_k = None
        return grad_A, grad_k

def setDominantSparseSymeig(A, Aadjoint_to_gadjoint):
    """
        Function primitive of dominant real symmetric eigensolver of a "sparse" matrix
    represented as a function.

        As a workaround of the fact that Pytorch doesn't support taking gradient 
    of objects of type other than torch.Tensor, the computation graph of this primitive 
    is slightly wrapped compared to DominantSymeig, which is the primitive of the case
    of the normal torch.Tensor representation, and is schematically shown below.
        ---------
        |     --|--> eigval
        |    /  |
    g --|-->A   |
        |    \  |
        |     --|--> eigvector
        ---------
    input: g -- The parameter(s) of interest of the matrix A, whose gradients are requested.
                In current version, g must be a torch.Tensor of arbitrary shape.
           k -- number of Lanczos vectors requested. (doesn't need gradient)
           dim -- The dimension of the square matrix A. (doesn't need gradient)
    output: eigval -- the smallest eigenvalue of A.
            eigvector -- corresponding (non-degenerate) eigenvector.
    
    USER NOTE: The mechanism of wrapping relies on user's providing two quantities:
        A -- The "sparse" representation of the matrix A as a function.
        Aadjoint_to_gadjoint -- A function that receive the adjoint of the matrix A
            as input, and return the adjoint of the pamameters(g) as output.

            The input should be of the form of two vectors represented as torch.Tensor, 
        say, v1 and v2, and the adjoint of A = v1 * v2^T.(outer product)
            User may do whatever he want to get the adjoint of g using these
        two vectors.
    """
    global DominantSparseSymeig 
    import CG_torch
    CG_torch.setCGSubspaceSparse(A, Aadjoint_to_gadjoint)
    @staticmethod
    def forward(ctx, g, k, dim):
        eigval, eigvector = symeigLanczos(A, k, extreme="min", sparse=True, dim=dim)
        ctx.save_for_backward(g, eigval, eigvector)
        return eigval, eigvector
    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvector):
        CG = CG_torch.CGSubspaceSparse.apply
        g, eigval, eigvector = ctx.saved_tensors
        b = grad_eigvector - torch.matmul(eigvector, grad_eigvector) * eigvector
        lambda0 = CG(g, eigval, b, eigvector)
        grad_A = grad_eigval * eigvector - lambda0, eigvector
        v1, v2 = grad_A
        grad_g = Aadjoint_to_gadjoint(v1, v2)
        grad_k = grad_dim = None
        return grad_g, grad_k, grad_dim
    DominantSparseSymeig = type("DominantSparseSymeig", (torch.autograd.Function, ), 
            {"forward": forward, "backward": backward})

if __name__ == "__main__":
    import time
    n = 1000
    A = 0.1 * torch.rand(n, n, dtype=torch.float64)
    A = A + A.T
    k = 300
    print("----- Test for Lanczos algorithm implemented using Pytorch -----")
    print("----- Dimension of real symmetric matrix A: %d -----" % n)

    start = time.time()
    eigval_min, eigvector_min, eigval_max, eigvector_max = symeigLanczos(A, k)
    end = time.time()
    print("Lanczos results:")
    print("lambda_min: ", eigval_min.item(), "lambda_max: ", eigval_max.item(), "running time: ", end - start)

    start = time.time()
    eigvals, eigvectors = torch.symeig(A, eigenvectors=True)
    end = time.time()
    print("Direct diagonalization results:")
    print("lambda_min: ", eigvals[0].item(), "lambda_max: ", eigvals[-1].item(), "running time: ", end - start)
    print("The difference between corresponding eigenvectors:")
    print("The 2-norm of the difference of v_mins corresponding to smallest eigenvalue lambda_min using two methods: ", 
            torch.norm(eigvector_min - eigvectors[:, 0]).item(),
            torch.norm(eigvector_min + eigvectors[:, 0]).item())
    print("The 2-norm of the difference of v_maxs corresponding to largest eigenvalue lambda_max using two methods: ", 
            torch.norm(eigvector_max - eigvectors[:, -1]).item(),
            torch.norm(eigvector_max + eigvectors[:, -1]).item())
