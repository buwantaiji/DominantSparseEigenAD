import torch

def CG_torch(A, b, initialx, sparse=False):
    """
        Compute the unique solution x of the system of linear equation Ax = b, 
    using Conjugate Gradient(CG) method and implemented by Pytorch.

    Input: 
        `A`: The square matrix A, which is assumed to be
            real symmetric and positive definite.
        `b`: The vector on the right hand side of the linear system.
        `initialx`: The initial vector of the CG algorithm.
            For certain cases, the initial vector should be properly chosen
            in order to get expected result, as is the case in backprop of
            diagonalization of real symmetric matrices by adjoint method.
        `sparse` indicates whether a bare linear function representation of the
            matrix is adopted. In any cases, the dimension of A is inferred
            from the size of the vector b.
    """
    if sparse:
        Amap = A
    else:
        Amap = lambda v: torch.matmul(A, v)
    n = b.shape[0]
    eps = 1e-7
    x = initialx
    r = b - Amap(x)
    if(torch.norm(r).item() < eps):
        return x
    d = r
    alpha = torch.matmul(r, r) / Amap(d).matmul(d)
    for i in range(n):
        x = x + alpha * d
        r_next = r - alpha * Amap(d)
        if(torch.norm(r_next).item() < eps):
            break
        beta = torch.matmul(r_next, r_next) / torch.matmul(r, r)
        r = r_next
        d = r + beta * d
        alpha = torch.matmul(r, r) / Amap(d).matmul(d)
    return x

def CGsubspace_forward(A, b, alpha, sparse=False):
    initialx = torch.randn(b.shape[0], dtype=b.dtype)
    initialx = initialx - torch.matmul(alpha, initialx) * alpha
    x = CG_torch(A, b, initialx, sparse=sparse)
    return x

def CGsubspace_backward(A, alpha, x, grad_x, sparse=False):
    initialx = torch.randn(x.shape[0], dtype=x.dtype)
    initialx = initialx - torch.matmul(alpha, initialx) * alpha
    b = grad_x - torch.matmul(alpha, grad_x) * alpha

    grad_b = CG_torch(A, b, initialx, sparse=sparse)
    if sparse:
        grad_A = - grad_b, x
    else:
        grad_A = - grad_b[:, None] * x
    grad_alpha = - x * torch.matmul(alpha, grad_x)
    return grad_A, grad_b, grad_alpha

def set_CGsubspace_sparse(H, Hadjoint_to_gadjoint):
    """
        Function primitive of low-rank CG linear system solver, where the matrix is
    "sparse" and represented as a function.

        As a workaround of the fact that Pytorch doesn't support taking gradient of
    objects of type other than torch.tensor, the computation graph of this primitive
    is wrapped compared to CGSubspace, which the version in which the matrix A is
    normally represented as a torch.Tensor. 
        In particular, this wrapped version is mainly used to make the back-propagation
    of the dominant sparse eigensolver primitive -- i.e., DominantSparseSymeig -- work
    properly. The computation graph is schematically shown below.
            -----------------
    g     --|--> A          |
            |     \         |
            |      A-E_0I --|--
            |     /         |  \ 
    E_0   --|-->--          |  |||--> x  
            |               |  / /
    b     --|------->-------|-- /
    alpha --|------->-------|---
            -----------------
    input: g -- The parameter(s) of interest of the matrix A, whose gradients are requested.
                In current version, g must be a torch.Tensor of arbitrary shape.
           E0, alpha are the smallest eigvalue and corresponding (non-degenerate)
                eigenvector, respectively.
    output: x.

        The computation process involves using CG algorithm to solve a low-rank linear
    system of the form (A - E_0I)x = b, alpha^T x = 0. For more details of this part, 
    c.f. https://buwantaiji.github.io/2019/10/CG-backward/

    USER NOTE: The mechanism of wrapping relies on user's providing two quantities:
        A -- The "sparse" representation of the matrix A as a function.
        Aadjoint_to_gadjoint -- A function that receive the adjoint of the matrix A
            as input, and return the adjoint of the pamameters(g) as output.

            The input should be of the form of two vectors represented as torch.Tensor, 
        say, v1 and v2, and the adjoint of A = v1 * v2^T.(outer product)
            User may do whatever he want to get the adjoint of g using these
        two vectors.
    """
    @staticmethod
    def forward(ctx, g, E0, b, alpha):
        A = lambda v: H(v) - E0 * v
        x = CGsubspace_forward(A, b, alpha, sparse=True)
        ctx.A = A
        ctx.save_for_backward(alpha, x)
        return x
    @staticmethod
    def backward(ctx, grad_x):
        A = ctx.A
        alpha, x = ctx.saved_tensors
        grad_A, grad_b, grad_alpha = CGsubspace_backward(A, alpha, x, grad_x, sparse=True)
        v1, v2 = grad_A
        grad_E0 = - torch.matmul(v1, v2)
        grad_g = Hadjoint_to_gadjoint(v1, v2)
        return grad_g, grad_E0, grad_b, grad_alpha
    global CGSubspaceSparse 
    CGSubspaceSparse = type("CGSubspaceSparse", (torch.autograd.Function, ), 
            {"forward": forward, "backward": backward})


class CGSubspace(torch.autograd.Function):
    """
        Function primitive of low-rank CG linear system solver, where the matrix is
    represented in normal form as a torch.Tensor.

    input: A, b, alpha, where A is a N-dimensional positive definite real symmetric
        matrix of rank N - 1, and alpha is the unique eigenvector of A of eigenvalue
        zero.(The other eigenvalues of A are all greater than zero.)
    output: the unique solution x of the low-rank linear system Ax = b in addition to
        the condition alpha^T x = 0.

    For details, c.f. https://buwantaiji.github.io/2019/10/CG-backward/
    """
    @staticmethod
    def forward(ctx, A, b, alpha):
        x = CGsubspace_forward(A, b, alpha)
        ctx.save_for_backward(A, alpha, x)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        A, alpha, x = ctx.saved_tensors
        return CGsubspace_backward(A, alpha, x, grad_x)

if __name__ == "__main__":
    """
    import numpy as np
    from scipy.stats import ortho_group
    n = 10
    diagonal = 1. + 10. * np.random.rand(n)
    U = ortho_group.rvs(n)
    # A is randomly generated as a real, symmetric, positive definite matrix
    #   of size n*n.
    A = U.dot(np.diag(diagonal)).dot(U.T)       
    A = torch.from_numpy(A).double()

    b = torch.randn(n, dtype=torch.float64)
    solution = torch.inverse(A).matmul(b)

    initialx = torch.randn(n, dtype=torch.float64)
    x = CG(A, b, initialx)
    print("solution:", solution)
    print("CG computation:", x)
    """
    n = 20
    A = torch.randn(n, n, dtype=torch.float64)
    A = A + A.T
    eigvalues, eigvectors = torch.symeig(A, eigenvectors=True)
    lambda0 = eigvalues[0]
    v0 = eigvectors[:, 0]
    Aprime = A - lambda0 * torch.eye(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)
    b = b - torch.matmul(v0, b) * v0
    initialx = torch.randn(n, dtype=torch.float64)
    initialx = initialx - torch.matmul(v0, initialx) * v0
    x = CG(Aprime, b, initialx)
    print("x = ", x)
    print("Aprime * x - b = ", torch.matmul(Aprime, x) - b)
