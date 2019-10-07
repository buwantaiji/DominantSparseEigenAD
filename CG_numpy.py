import numpy as np

def CG(A, b, initialx):
    """
        Compute the unique solution x of the system of linear equation Ax = b, 
    using Conjugate Gradient(CG) method and implemented by Numpy.
    The square matrix A is assumed to be real symmetric and positive definite.
    """
    n = b.size
    eps = 1e-7
    x = initialx
    #print("i=0(initialization):", x)
    r = b - np.dot(A, x)
    if(np.linalg.norm(r) < eps):
        return x
    d = r
    alpha = np.dot(r, r) / d.dot(A).dot(d)
    for i in range(n):
        x = x + alpha * d
        #print("i=%d:" % (i + 1), x)
        r_next = r - alpha * np.dot(A, d)
        if(np.linalg.norm(r_next) < eps):
            break
        beta = np.dot(r_next, r_next) / np.dot(r, r)
        r = r_next
        d = r + beta * d
        alpha = np.dot(r, r) / d.dot(A).dot(d)
    return x

if __name__ == "__main__":
    """
    from scipy.stats import ortho_group
    n = 10
    diagonal = 1. + 10. * np.random.rand(n)
    U = ortho_group.rvs(n)
    A = U.dot(np.diag(diagonal)).dot(U.T)       # A is randomly generated as a real, symmetric, positive definite matrix of size n*n.
    b = np.random.randn(n)
    solution = np.linalg.inv(A).dot(b)
    x = CG(A, b, np.random.randn(n))
    print("solution:", solution)
    print("CG computation:", x)
    """
    n = 10
    A = np.random.randn(n, n)
    A = A + A.T
    eigvalues, eigvectors = np.linalg.eigh(A)
    lambda0 = eigvalues[0]
    v0 = eigvectors[:, 0]
    Aprime = A - lambda0 * np.eye(n)
    b = np.random.randn(n)
    b = b - np.dot(v0, b) * v0
    initialx = np.random.randn(n)
    initialx = initialx - np.dot(v0, initialx) * v0
    x = CG(Aprime, b, initialx)
    print("x = ", x)
    print("Aprime * x - b = ", np.dot(Aprime, x) - b)
