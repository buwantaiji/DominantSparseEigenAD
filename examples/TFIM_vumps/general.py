"""
    Variational MPS optimization of 1D Transverse Field Ising Model(TFIM).
In this implementation, A in the MPS is a general, real rank-3 tensor without any
symmetry presumptions.
"""
import numpy as np
import torch
from DominantSparseEigenAD.GeneralLanczos import DominantEig

class TFIM(torch.nn.Module):
    def __init__(self, D, k):
        super(TFIM, self).__init__()
        self.d = 2
        self.D = D
        self.k = k
        print("----- MPS representation of 1D TFIM -----")
        print("General setting. D =", self.D)
        self.dominant_eig = DominantEig.apply
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
        self.A = torch.nn.Parameter(A)

    def forward(self):
        Gong = torch.einsum("kij,kmn->imjn", self.A, self.A).reshape(self.D**2, self.D**2)
        eigval_max, leigvector_max, reigvector_max = self.dominant_eig(Gong, self.k)
        leigvector_max = leigvector_max.reshape(self.D, self.D)
        reigvector_max = reigvector_max.reshape(self.D, self.D)
        E0 = torch.einsum("aik,bkj,abcd,cml,dln,im,jn", self.A, self.A, self.h, 
                self.A, self.A, leigvector_max, reigvector_max) / eigval_max**2
        return E0

if __name__ == "__main__":
    D = 20
    k = 200
    model = TFIM(D, k)
    data_E0 = np.load("TFIM/datas/E0_N_100000.npz")
    gs = data_E0["gs"]
    E0s = data_E0["E0s"]

    def closure():
        E0 = model()
        optimizer.zero_grad()
        E0.backward()
        return E0

    for i in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
        model.seth(gs[i])
        model.setparameters()
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=10, tolerance_grad=1E-7)
        print("g =", gs[i], "E0 =", E0s[i])
        iter_num = 50
        for i in range(iter_num):
            E0 = optimizer.step(closure)
            print("iter: ", i, E0.item())
