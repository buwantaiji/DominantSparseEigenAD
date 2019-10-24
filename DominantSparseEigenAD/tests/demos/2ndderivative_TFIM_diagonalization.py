import torch
import numpy as np

class TFIM(object):
    def __init__(self, N):
        self.N = N
        self.dim = 2**N
        print("Lattice size N = %d" % self.N)
        self._diags()
        self._flips_basis()
        print("Model initialization completed.")

    def _diags(self):
        indices = np.arange(self.dim)[:, np.newaxis]
        bin_reps = (indices >> np.arange(self.N)[::-1]) & 1
        spins = 1 - 2 * bin_reps
        spins_prime = np.hstack( (spins[:, 1:], spins[:, 0:1]) )
        self.diag_elements = - (spins * spins_prime).sum(axis=1)
        self.diag_elements = torch.from_numpy(self.diag_elements).to(torch.float64)

    def _flips_basis(self):
        masks = torch.Tensor([1 << i for i in range(self.N)]).long()
        basis = torch.arange(self.dim)[:, None]
        self.flips_basis = basis ^ masks

    def setHmatrix(self):
        diagmatrix = torch.diag(self.diag_elements)
        offdiagmatrix = torch.zeros(self.dim, self.dim).to(torch.float64)
        offdiagmatrix[self.flips_basis.T, torch.arange(self.dim)] = 1.0
        randommatrix = 1e-12 * torch.randn(model.dim, model.dim).to(torch.float64)
        randommatrix = 0.5 * (randommatrix + randommatrix.T)

        self.Hmatrix = diagmatrix - self.g * offdiagmatrix + randommatrix

if __name__ == "__main__":
    N = 6
    model = TFIM(N)

    for g in np.linspace(0.0, 2.0, num=21):
        model.g = torch.Tensor([g]).to(torch.float64)
        model.g.requires_grad_(True)

        model.setHmatrix()

        Es, psis = torch.symeig(model.Hmatrix, eigenvectors=True)
        E0 = Es[0]
        dE0, = torch.autograd.grad(E0, model.g, create_graph=True)
        d2E0, = torch.autograd.grad(dE0, model.g)

        print(g, E0.item() / model.N, 
                        dE0.item() / model.N, 
                        d2E0.item() / model.N)
