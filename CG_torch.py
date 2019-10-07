import torch

def CG(A, b, initialx, sparse=False):
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
