import numpy as np

def Lanczos(A, k):
    """
    Lanczos iteration algorithm on a real symmetric matrix using numpy.
    Input: A is a real symmetrix matrix of size, say, n.
           k is the number of Lanczos vector requested. In typical applications, k << n.
    Output: A tuple (Qk, T), where Qk = (q1 q2 ... qk) is a n*k matrix, whose columns contain k orthomormal Lanczos vectors q1, q2, ..., qk.
            I.e., we have Qk^T * Qk = I_k, where I_k is the k-dimensional identity matrix.
            T is a tridiagonal matrix of size k.
        Theoretically, when input k = n, the corresponding outputs Qn and T satisfy Qn^T * A * Q = T, and the eigenvalues and eigenvectors of T will be
    identically the same as that of original matrix A. However, for any k = 1, 2, ..., n, the Lanczos vectors q1, q2, ..., qk are carefully selected
    in this algorithm that they constitute a orthonormal basis of Krylov space K(A, q1, k) = span{q1, A*q1, ..., A^(k-1)*q1}. Then it can be shown
    that the eigenvalues in the extreme region(i.e., closed to largest and smallest eigenvalues) and corresponding eigenvectors can be accurately
    approximated by the results obtained by solving the original eigenvalue problem restricted in the Krylov subspace K(A, q1, k), even though its
    dimension k (the number of iterations actually performed) is FAR LESS THAN n.
        In practice, after this subroutine is called, one can diagonalize the tridiagonal matrix T to get accurate approximations of eigenvalues of A in
    the extreme region. The corresponding eigenvectors are obtained by multiplying the "eigenvector representation in the k-dimensional Krylov subspace",
    which is a k-dimensional vector, by the matrix Qk.
        In practice, the Lanczos iteration turns out to be unstable upon floating point arithmetic. The basic difficulty is caused by loss of orthogonality
    among the Lanczos vectors, which would mess up the actual result of eigenvectors. In current version, the simple but a bit expensive "full reorthogonalization"
    approach is adopted to cure this problem.
    """
    n = A.shape[0]
    Qk = np.zeros((n, k))
    alphas = np.zeros(k)
    betas = np.zeros(k - 1)
    q = np.random.randn(n)
    q = q / np.linalg.norm(q)
    u = np.dot(A, q)
    alpha = np.dot(q, u)
    Qk[:, 0] = q
    alphas[0] = alpha
    beta = 0
    qprime = np.random.randn(n)
    for i in range(1, k):
        r = u - alpha * q - beta * qprime

        # The simple but expensive full reorthogonalization process in order to recover the orthogonality among the Lanczos vectors caused by
        # rounding error in floating point arithmetic.
        r -= np.dot(Qk[:, :i], np.dot(Qk[:, :i].T, r))

        qprime = q
        beta = np.linalg.norm(r)
        q = r / beta
        u = np.dot(A, q)
        alpha = np.dot(q, u)
        alphas[i] = alpha
        betas[i - 1] = beta
        Qk[:, i] = q
    T = np.diag(alphas) + np.diag(betas, k=1) + np.diag(betas, k=-1)
    return Qk, T

if __name__ == "__main__":
    import time
    n = 5000
    A = 0.1 * np.random.rand(n, n)
    A = A + A.T
    k = 300
    print("----- Test for Lanczos algorithm implemented using Numpy -----")
    print("----- Dimension of real symmetric matrix A: %d -----" % n)

    start = time.time()
    Qk, T = Lanczos(A, k)
    #print("Qk^T * Qk = ", np.dot(Qk.T, Qk))
    eigvalsQ, eigvectorsQ = np.linalg.eigh(T)
    #print("eigvectorsQ.T * eigvectorsQ = ", np.dot(eigvectorsQ.T, eigvectorsQ))
    eigvectorsQ = np.dot(Qk, eigvectorsQ)
    end = time.time()
    print("Lanczos results:")
    print("lambda_min: ", eigvalsQ[0], "lambda_max: ", eigvalsQ[-1], "running time: ", end - start)

    start = time.time()
    eigvals, eigvectors = np.linalg.eigh(A)
    end = time.time()
    print("Direct diagonalization results:")
    print("lambda_min: ", eigvals[0], "lambda_max: ", eigvals[-1], "running time: ", end - start)
    print("The difference between corresponding eigenvectors:")
    print("The 2-norm of the difference of v_mins corresponding to smallest eigenvalue lambda_min using two methods: ", 
            np.linalg.norm(eigvectorsQ[:, 0] - eigvectors[:, 0]),
            np.linalg.norm(eigvectorsQ[:, 0] + eigvectors[:, 0]))
    print("The 2-norm of the difference of v_maxs corresponding to largest eigenvalue lambda_max using two methods: ", 
            np.linalg.norm(eigvectorsQ[:, -1] - eigvectors[:, -1]),
            np.linalg.norm(eigvectorsQ[:, -1] + eigvectors[:, -1]))
