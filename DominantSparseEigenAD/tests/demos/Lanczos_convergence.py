"""
    Demonstration that the Lanczos algorithm cannot be used effectively for some
kinds of matrices.
    In the code, the function H2 generates a random real symmetric matrix, which
Lanczos algorithm can diagonalize effectively using a small number of k. 
    On the other hand, the function H1 generates a tridiagonal real symmetric matrix, 
which arises naturally in solving 1D schrodinger equation in coordinate space.
It turns out that the Lanczos method can yield satisfying results only when k
reaches the maximal possible value, i.e., the dimension of the matrix.
"""
import numpy as np
import torch
from DominantSparseEigenAD.Lanczos import symeigLanczos
def H1(N):
    xmin, xmax = -1., 1.
    xmesh = np.linspace(xmin, xmax, num=N, endpoint=False)
    xmesh = torch.from_numpy(xmesh).to(torch.float64)
    h = (xmax - xmin) / N
    K = -0.5/h**2 * (torch.diag(-2 * torch.ones(N, dtype=xmesh.dtype))
                    + torch.diag(torch.ones(N - 1, dtype=xmesh.dtype), diagonal=1)
                    + torch.diag(torch.ones(N - 1, dtype=xmesh.dtype), diagonal=-1))
    potential = 0.5 * xmesh**2
    V = torch.diag(potential)
    Hmatrix = K + V
    return Hmatrix

def H2(N):
    Hmatrix = 0.1 * torch.rand(N, N, dtype=torch.float64)
    Hmatrix = Hmatrix + Hmatrix.T
    return Hmatrix

if __name__ == "__main__":
    N = 300

    ks = np.arange(10, N + 1, step=2)
    Hmatrix = H1(N)
    #ks = np.arange(10, N + 1, step=2)
    #Hmatrix = H2(N)

    E0s, _ = torch.symeig(Hmatrix, eigenvectors=True)
    E0_groundtruth = E0s[0].item()
    #print("Groundtruth: ", E0s[0].item())
    E0s_lanczos = np.empty(ks.size)
    relative_error = np.empty(ks.size)
    for i in range(ks.size):
        E0s_lanczos[i], _ = symeigLanczos(Hmatrix, ks[i], extreme="min")
        relative_error[i] = np.log10( np.abs(E0s_lanczos[i] - E0_groundtruth) \
                            / np.abs(E0_groundtruth) )
        print("k = ", ks[i], relative_error[i])

    import matplotlib.pyplot as plt
    plt.plot(ks, relative_error)
    plt.title("Log relative error of the minimum eigenvalue using various numbers of Lanczos vectors $k$\n"
            "Dimension of the matrix being diagonalized: %d" % N)
    plt.xlabel("$k$")
    plt.ylabel("Log relative error")
    plt.show()
