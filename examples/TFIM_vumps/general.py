"""
    Variational MPS optimization of 1D Transverse Field Ising Model(TFIM).
In this implementation, A in the MPS is a general, real rank-3 tensor without any
symmetry presumptions.
"""
import numpy as np
import scipy.sparse.linalg as sparselinalg
import torch
from DominantSparseEigenAD.eig import DominantEig
import DominantSparseEigenAD.eig as eig

class TFIM(torch.nn.Module):
    def __init__(self, D, k):
        super(TFIM, self).__init__()
        self.d = 2
        self.D = D
        self.k = k
        print("----- MPS representation of 1D TFIM -----")
        print("General setting. D = %d, k = %d." % (self.D, self.k))
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

    def setparameters(self, initA=None):
        #A = torch.randn(self.d, self.D, self.D, dtype=torch.float64) if initA is None else initA
        if initA is None:
            A = torch.randn(self.d, self.D, self.D, dtype=torch.float64)
            print("Random initialization.")
        else:
            A = initA
            print("Initialization using the last optimized result.")
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
        eig.setDominantSparseEig(self.Gong, self.GongT, self.Gongadjoint_to_Aadjoint)
        dominant_sparse_eig = eig.DominantSparseEig.apply 
        eigval_max, leigvector_max, reigvector_max = dominant_sparse_eig(self.A, self.k)
        leigvector_max = leigvector_max.reshape(self.D, self.D)
        reigvector_max = reigvector_max.reshape(self.D, self.D)
        E0 = self._h_optcontraction(leigvector_max, reigvector_max) / eigval_max**2
        return E0

if __name__ == "__main__":
    import time
    data_E0 = np.load("datas/E0_sum.npz")
    gs = data_E0["gs"]
    E0s = data_E0["E0s"]

    for g_idx in [3, 4, 5]:
        g, E0 = gs[g_idx], E0s[g_idx]
        print("g = %f, E0 = %.15f" % (g, E0))

        Ds = np.arange(5, 100+1, step=5)
        ks = np.array([10, 50, 80, 80, 100, 100] + [200] * 14)
        Npoints = Ds.size
        E0s_general = np.empty(Npoints)

        def closure():
            #E0 = model.matrix_forward()
            E0 = model.sparse_forward()
            optimizer.zero_grad()
            E0.backward()
            return E0

        initA = None
        for i in range(Npoints):
            D, k = Ds[i], ks[i]
            model = TFIM(D, k)
            model.seth(g)
            model.setparameters(initA=initA)
            optimizer = torch.optim.LBFGS(model.parameters(), max_iter=20,
                tolerance_grad=0.0, tolerance_change=0.0, line_search_fn="strong_wolfe")
            iter_num = 60
            for epoch in range(iter_num):
                start = time.time()
                E0 = optimizer.step(closure)
                end = time.time()
                print("iter: ", epoch, E0.item(), end - start)
                #print(model.A.detach().max().item(), model.A.grad.max().item())
                if epoch == iter_num - 1:
                    E0s_general[i] = E0.detach().numpy()
                    if i != Npoints - 1:
                        Dnew = Ds[i+1]
                        initA = 1. * torch.randn(model.d, Dnew, Dnew, dtype=torch.float64)
                        initA[:, :D, :D] = model.A.detach()

        for i in range(Npoints):
            print("D = %d: \t%.15f" % (Ds[i], E0s_general[i]))

        filename = "datas/E0s_general/g_%.2f.npz" % g
        np.savez(filename, Ds=Ds, E0s=E0s_general)
