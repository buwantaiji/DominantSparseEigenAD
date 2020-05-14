import torch

def Lanczos(A, k, device=torch.device("cpu"), *, sparse=False, dim=None):
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
    Qk = torch.zeros((n, k), dtype=dtype, device=device)
    alphas = torch.zeros(k, dtype=dtype, device=device)
    betas = torch.zeros(k - 1, dtype=dtype, device=device)
    q = torch.randn(n, dtype=dtype, device=device)
    q = q / torch.norm(q)
    u = Amap(q)
    alpha = torch.matmul(q, u)
    Qk[:, 0] = q
    alphas[0] = alpha
    beta = 0
    qprime = torch.randn(n, dtype=dtype, device=device)
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

def symeigLanczos(A, k, device=torch.device("cpu"), extreme="both", *, 
                    sparse=False, dim=None):
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
    Qk, T = Lanczos(A, k, device=device, sparse=sparse, dim=dim)
    eigvalsQ, eigvectorsQ = torch.symeig(T, eigenvectors=True)
    eigvectorsQ = torch.matmul(Qk, eigvectorsQ)
    if extreme == "both":
        return eigvalsQ[0], eigvectorsQ[:, 0], eigvalsQ[-1], eigvectorsQ[:, -1]
    elif extreme == "min":
        return eigvalsQ[0], eigvectorsQ[:, 0]
    elif extreme == "max":
        return eigvalsQ[-1], eigvectorsQ[:, -1]
