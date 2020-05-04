"""
    Variational MPS optimization of 1D Transverse Field Ising Model(TFIM).
In this implementation, A in the MPS is a general, real rank-3 tensor without any
symmetry presumptions.
"""
import numpy as np
import scipy.sparse.linalg as sparselinalg
import torch
from DominantSparseEigenAD.GeneralLanczos import DominantEig
import DominantSparseEigenAD.GeneralLanczos as generallanczos

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

    def matrix_forward(self):
        """
            Forward pass by explicitly constructing the transfer matrix "Gong"
        as a torch.Tensor.
        """
        Gong = torch.einsum("kij,kmn->imjn", self.A, self.A).reshape(self.D**2, self.D**2)
        eigval_max, leigvector_max, reigvector_max = self.dominant_eig(Gong, self.k)
        leigvector_max = leigvector_max.reshape(self.D, self.D)
        reigvector_max = reigvector_max.reshape(self.D, self.D)
        E0 = torch.einsum("aik,bkj,abcd,cml,dln,im,jn", self.A, self.A, self.h, 
                self.A, self.A, leigvector_max, reigvector_max) / eigval_max**2
        return E0

    def _setsparsefunctions(self):
        A = self.A.detach().numpy()
        def fr(v):
            r = v.reshape(self.D, self.D)
            return np.einsum("kij,kmn,jn->im", A, A, r, optimize="greedy")
        self.Gong = sparselinalg.LinearOperator((self.D**2, self.D**2), matvec=fr)
        def fl(v):
            l = v.reshape(self.D, self.D)
            return np.einsum("kij,kmn,im->jn", A, A, l, optimize="greedy")
        self.GongT = sparselinalg.LinearOperator((self.D**2, self.D**2), matvec=fl)
        def Gongadjoint_to_Aadjoint(grad_Gong):
            grad_A = np.zeros((self.d, self.D, self.D))
            for u, v in grad_Gong:
                umat, vmat = u.reshape(self.D, self.D), v.reshape(self.D, self.D)
                grad_A = grad_A \
                    + np.einsum("im,jn,kmn->kij", umat, vmat, A, optimize="greedy") \
                    + np.einsum("mi,nj,kmn->kij", umat, vmat, A, optimize="greedy")
            return torch.from_numpy(grad_A)
        self.Gongadjoint_to_Aadjoint = Gongadjoint_to_Aadjoint

    def _h_optcontraction(self, l, r):
        upperleft = torch.einsum("aik,im->amk", self.A, l)
        upperright = torch.einsum("bkj,jn->bkn", self.A, r)
        upper = torch.einsum("amk,bkn->abmn", upperleft, upperright)
        lower = torch.einsum("cml,dln->cdmn", self.A, self.A)
        upperlower = torch.einsum("abmn,cdmn->abcd", upper, lower)
        result = torch.einsum("abcd,abcd", upperlower, self.h)
        return result

    def sparse_forward(self):
        """
            Forward pass by treating the transfer matrix "Gong" as a "sparse matrix"
        represented by scipy.sparse.linalg.LinearOperator. This way, various tensor
        contractions involved in the forward and backward pass can be significantly
        optimized and accelerated.
        """
        self._setsparsefunctions()
        generallanczos.setDominantSparseEig(self.Gong, self.GongT,
                self.Gongadjoint_to_Aadjoint)
        dominant_sparse_eig = generallanczos.DominantSparseEig.apply
        eigval_max, leigvector_max, reigvector_max = dominant_sparse_eig(self.A, self.k)
        leigvector_max = leigvector_max.reshape(self.D, self.D)
        reigvector_max = reigvector_max.reshape(self.D, self.D)
        E0 = self._h_optcontraction(leigvector_max, reigvector_max) / eigval_max**2
        return E0

if __name__ == "__main__":
    import time
    D = 100
    k = 200
    model = TFIM(D, k)
    data_E0 = np.load("datas/E0_sum.npz")
    gs = data_E0["gs"]
    E0s = data_E0["E0s"]
    Npoints = gs.size
    E0s_general = np.empty(Npoints)

    def closure():
        #E0 = model.matrix_forward()
        E0 = model.sparse_forward()
        optimizer.zero_grad()
        E0.backward()
        return E0

    for i in range(Npoints):
        model.seth(gs[i])
        model.setparameters()
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=20,
            tolerance_grad=0.0, tolerance_change=0.0, line_search_fn="strong_wolfe")
        print("g = %f, E0 = %.15f" % (gs[i], E0s[i]))
        iter_num = 100
        for epoch in range(iter_num):
            start = time.time()
            E0 = optimizer.step(closure)
            end = time.time()
            print("iter: ", epoch, E0.item(), end - start)
            if epoch == iter_num - 1:
                E0s_general[i] = E0.detach().numpy()

    for i in range(Npoints):
        print("%f: %.15f, \t%.15f" % (gs[i], E0s[i], E0s_general[i]))

    filename = "datas/E0s_general1/D_%d.npz" % D
    np.savez(filename, gs=gs, E0s=E0s_general)
