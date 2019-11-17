"""
    Demonstration of the significant performance difference of different contraction
orders in Tensor network applications.
"""
import numpy as np
import torch
import time

def contraction_numpy(d, D):
    print("----- Numpy -----")
    A = np.random.randn(d, D, D)

    start = time.time()
    Gong = np.einsum("kij,kmn->imjn", A, A).reshape(D**2, D**2)
    end = time.time()
    print("constructig Gong: (~ D^4 * d)\t", end - start)

    r = np.random.randn(D, D)
    r_flat = r.reshape(D**2)

    #path_info = np.einsum_path("kij,kmn,jn->im", A, A, r)
    #print(path_info[0])

    start = time.time()
    result1 = Gong.dot(r_flat)
    end = time.time()
    print("method1: (~ D^4)\t\t", end - start)

    start = time.time()
    result2 = np.einsum("kij,kmn,jn->im", A, A, r, optimize="greedy").reshape(D**2)
    end = time.time()
    print("method2: (~D^3 * d)\t\t", end - start)

    assert np.allclose(result1, result2)

def contraction_torch(d, D):
    print("----- Pytorch -----")
    A = torch.randn(d, D, D, dtype=torch.float64)

    start = time.time()
    Gong = torch.einsum("kij,kmn->imjn", A, A).reshape(D**2, D**2)
    end = time.time()
    print("constructig Gong: (~ D^4 * d)\t", end - start)

    r = torch.randn(D, D, dtype=torch.float64)
    r_flat = r.reshape(D**2)

    start = time.time()
    """
        method1: matrix multiplication.
        ~ D^4
    """
    result1 = Gong.matmul(r_flat)
    end = time.time()
    print("method1: (~ D^4)\t\t", end - start)

    start = time.time()
    """
        method2: (manual) optimized einsum.
        ~ D^3 * d
    """
    intermediate = torch.einsum("kij,jn->kin", A, r)
    result2 = torch.einsum("kin,kmn->im", intermediate, A).reshape(D**2)
    end = time.time()
    print("method2: (~D^3 * d)\t\t", end - start)

    start = time.time()
    """
        method3: native torch einsum (not optimized).
        ~ D^4 * d + D^4. i.e., roughly equal to the cost of the construction of the 
    matrix Gong and the matrix multiplication process(method 1).
    """
    result3 = torch.einsum("kij,kmn,jn->im", A, A, r).reshape(D**2)
    end = time.time()
    print("method3: (~ D^4 * d + D^4)\t", end - start)

    assert torch.allclose(result1, result2)
    assert torch.allclose(result1, result3)

if __name__ == "__main__":
    d = 2
    D = 80
    print("d = %d, D = %d" % (d, D))
    contraction_numpy(d, D)
    contraction_torch(d, D)
