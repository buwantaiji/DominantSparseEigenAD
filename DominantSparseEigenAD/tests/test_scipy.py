import time, pytest
import numpy as np
from scipy import linalg
from scipy.sparse import linalg as sparselinalg

def test_eigs():
    N = 1000
    k = 5
    ncv = 200
    A = np.random.randn(N, N)
    print("\n----- test_eigs -----")
    print("----- Dimension of matrix A: %d -----" % N)
    print("scipy.sparse.linalg.eigs time: ")

    start = time.time()
    righteigvals, righteigvectors = sparselinalg.eigs(A, k, ncv=ncv)
    end = time.time()
    print("right eigens: ", end - start)
    sortidx = (righteigvals.conj() * righteigvals).real.argsort()
    righteigvals = righteigvals[sortidx]
    righteigvectors = righteigvectors[:, sortidx]

    start = time.time()
    lefteigvals, lefteigvectors = sparselinalg.eigs(A.T, k, ncv=ncv)
    end = time.time()
    sortidx = (lefteigvals.conj() * lefteigvals).real.argsort()
    lefteigvals = lefteigvals[sortidx]
    lefteigvectors = lefteigvectors[:, sortidx]
    print("left eigens: ", end - start)

    assert np.allclose(lefteigvals, righteigvals)
    orthogonals = lefteigvectors.T.dot(righteigvectors)
    mask = np.ones((k, k)) - np.eye(k)
    assert np.allclose(orthogonals * mask, np.zeros((k, k)))

    assert np.allclose((righteigvectors.conj() * righteigvectors).sum(axis=0), 
            np.ones(k))
    assert np.allclose((lefteigvectors.conj() * lefteigvectors).sum(axis=0), 
            np.ones(k))

def test_Gong():
    D = 20
    d = 2
    #A = np.random.randn(d, D, D) + 1.j * np.random.randn(d, D, D)
    A = np.random.randn(d, D, D)
    Gong = np.einsum("kij,kmn->imjn", A, A.conj()).reshape(D**2, D**2)

    righteigvals, righteigvectors = sparselinalg.eigs(Gong, k=5, ncv=100)
    print("\n", righteigvals)
    print(righteigvals.conj() * righteigvals)
    maxidx = (righteigvals.conj() * righteigvals).real.argmax()
    print("maxidx =", maxidx)
    #print(righteigvectors[:, maxidx])
    assert np.allclose(righteigvals[maxidx].imag, 0.0)
    assert np.allclose(righteigvectors[:, maxidx].imag, np.zeros(D**2))

    lefteigvals, lefteigvectors = sparselinalg.eigs(Gong.T, k=5, ncv=100)
    print("\n", lefteigvals)
    print(lefteigvals.conj() * lefteigvals)
    maxidx = (lefteigvals.conj() * lefteigvals).real.argmax()
    print("maxidx =", maxidx)
    #print(lefteigvectors[:, maxidx])
    assert np.allclose(lefteigvals[maxidx].imag, 0.0)
    assert np.allclose(lefteigvectors[:, maxidx].imag, np.zeros(D**2))

def test_eigsh():
    N = 1000
    k = 5
    ncv = 100
    A = np.random.randn(N, N)
    A = 0.5 * (A + A.T)
    print("\n----- test_eigsh -----")
    print("----- Dimension of real symmetric matrix A: %d -----" % N)

    start = time.time()
    eigvals_full, eigvectors_full = linalg.eigh(A)
    end = time.time()
    print("scipy.linalg.eigh time: ", end - start)

    start = time.time()
    eigvals, eigvectors = sparselinalg.eigsh(A, k, which="SA", ncv=ncv)
    end = time.time()
    print("scipy.sparse.linalg.eigsh time: ", end - start)
    assert np.allclose(eigvals, eigvals_full[:k])
    for i in range(k):
        assert np.allclose(eigvectors[:, i], eigvectors_full[:, i]) or \
                np.allclose(eigvectors[:, i], -eigvectors_full[:, i])

@pytest.mark.skip(reason="Incorrect behavior of the scipy sparse linear system "
        "solvers when the matrix dimension is large.")
def test_linsys_fullrank():
    #from krypy.linsys import LinearSystem, Gmres
    N = 20
    A = np.random.randn(N, N)
    b = np.random.randn(N)

    print("\n----- test_linsys_fullrank -----")
    print("----- Dimension of matrix A: %d -----" % N)
    #linear_system = LinearSystem(A, b)
    #solver = Gmres(linear_system)
    #x_krypy = solver.xk[:, 0]
    #print("Krypy gmres time: ", end - start)
    #assert np.allclose(A.dot(x_krypy), b)
    start = time.time()
    x, code = sparselinalg.gmres(A, b, tol=1e-12, atol=1e-12)
    end = time.time()
    print("code: ", code)
    print("scipy gmres time: ", end - start)
    assert np.allclose(A.dot(x), b)

def test_linsys_lowrank():
    N = 1000
    A = np.random.randn(N, N)
    ncv = 100
    print("\n----- test_linsys_lowrank -----")
    print("----- Dimension of matrix A: %d -----" % N)

    righteigval, righteigvector = sparselinalg.eigs(A, k=1, ncv=ncv, which="LM")
    lefteigval, lefteigvector = sparselinalg.eigs(A.T, k=1, ncv=ncv, which="LM")
    assert np.allclose(righteigval, lefteigval)
    eigval = righteigval
    righteigvector = righteigvector[:, 0]
    lefteigvector = lefteigvector[:, 0]
    lefteigvector /= np.dot(lefteigvector, righteigvector)

    print("scipy gmres time: ")
    Aprime = A - eigval * np.eye(N)
    b = np.random.randn(N)
    b = b - righteigvector * np.dot(lefteigvector, b)
    start = time.time()
    x, code = sparselinalg.gmres(Aprime, b, tol=1e-12, atol=1e-12)
    end = time.time()
    print(end - start)
    assert np.allclose(Aprime.dot(x), b)
    assert np.allclose(np.dot(lefteigvector, x), 0.0)

    ATprime = A.T - eigval * np.eye(N)
    b = np.random.randn(N)
    b = b - lefteigvector * np.dot(righteigvector, b)
    start = time.time()
    x, code = sparselinalg.gmres(ATprime, b, tol=1e-12, atol=1e-12)
    end = time.time()
    print(end - start)
    assert np.allclose(ATprime.dot(x), b)
    assert np.allclose(np.dot(righteigvector, x), 0.0)
