"""
    Variational MPS optimization of 1D Transverse Field Ising Model(TFIM).
In this implementation, A in the MPS is a real rank-3 tensor that is symmetric
under permutation of the two virtual indices. As a result, the transfer matrix
is symmetric.
    Note that this implementation is somewhat deprecated regarding this symmetry
restriction. For a more complete implementation, see the file general.py.
"""
import numpy as np
import torch
from DominantSparseEigenAD.symeig import DominantSymeig

class TFIM(torch.nn.Module):
    def __init__(self, D, k):
        super(TFIM, self).__init__()
        self.d = 2
        self.D = D
        self.k = k
        print("----- MPS representation of 1D TFIM -----")
        print("Symmetric setting. D =", self.D)
        self.dominant_symeig = DominantSymeig.apply
    def seth(self, g):
        """
            Construct the nearest neighbor hamiltonian for given parameter g of
        the 1D TFIM.
        """
        self.h = torch.zeros(self.d, self.d, self.d, self.d, dtype=torch.float64)
        self.h[0, 0, 0, 0] = self.h[1, 1, 1, 1] = -1.0
        self.h[0, 1, 0, 1] = self.h[1, 0, 1, 0] = 1.0
        self.h[1, 0, 0, 0] = self.h[0, 1, 0, 0] = \
        self.h[1, 1, 0, 1] = self.h[0, 0, 0, 1] = \
        self.h[0, 0, 1, 0] = self.h[1, 1, 1, 0] = \
        self.h[0, 1, 1, 1] = self.h[1, 0, 1, 1] = -g/2

    def setparameters(self):
        A = torch.randn(self.d, self.D, self.D, dtype=torch.float64)
        A = 0.5 * (A + A.permute(0, 2, 1))
        self.A = torch.nn.Parameter(A)

    def forward(self):
        A = 0.5 * (self.A + self.A.permute(0, 2, 1))
        Gong = torch.einsum("kij,kmn->imjn", A, A).reshape(self.D**2, self.D**2)
        minus_Gong = - Gong
        eigval_max, eigvector_max = self.dominant_symeig(minus_Gong, self.k)
        eigvector_max = eigvector_max.reshape(self.D, self.D)
        E0 = torch.einsum("aik,bkj,abcd,cml,dln,im,jn", A, A, self.h, 
                A, A, eigvector_max, eigvector_max) / eigval_max**2
        return E0

if __name__ == "__main__":
    D = 20
    k = 100
    model = TFIM(D, k)
    data_E0 = np.load("datas/E0_sum.npz")
    gs = data_E0["gs"]
    E0s = data_E0["E0s"]

    def closure():
        E0 = model()
        optimizer.zero_grad()
        E0.backward()
        return E0

    for i in [3, 4, 5]:
        model.seth(gs[i])
        model.setparameters()
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=10, tolerance_grad=1E-7)
        print("g =", gs[i], "E0 =", E0s[i])
        iter_num = 50
        for i in range(iter_num):
            E0 = optimizer.step(closure)
            print("iter: ", i, E0.item())
