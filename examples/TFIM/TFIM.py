import numpy as np
import torch

class TFIM(object):
    """
        Direct diagonalization solver of 1D Transverse Field Ising Model(TFIM).
    H = - \sum_{i=0}^{N-1} (g\sigma_i^x + \sigma_i^z \sigma_{i+1}^z)
    """
    def __init__(self, N, device=torch.device("cpu")):
        self.N = N
        self.dim = 2**N
        print("Lattice size N = %d" % self.N)
        self.device = device
        self._diags()
        self._flips_basis()
        print("Model initialization completed.")

    def analytic_results(self):
        """
            Some analytic results of the model, based on Jordan-Wigner transformation.
        The formulas may be a little problematic for finite lattice size N.

        E0_per_N:      E0 / N
        pE0_per_N_pg:  \partial (E0 / N) / \partial g
        p2E0_per_N_pg2: \partial2 (E0 / N) / \partial g2
        """
        g = self.g.detach().item()
        ks = np.linspace(-(self.N-1)/2, (self.N-1)/2, num=N) / self.N * 2 * np.pi

        epsilon_ks = 2 * np.sqrt(g**2 - 2 * g * np.cos(ks) + 1)
        pepsilon_ks_pg = 4 * (g - np.cos(ks)) / epsilon_ks
        p2epsilon_ks_pg2 = 16 * np.sin(ks)**2 / epsilon_ks**3

        E0_per_N = - 0.5 * epsilon_ks.sum() / self.N
        pE0_per_N_pg = - 0.5 * pepsilon_ks_pg.sum() / self.N
        p2E0_per_N_pg2 = - 0.5 * p2epsilon_ks_pg2.sum() / self.N
        return E0_per_N, pE0_per_N_pg, p2E0_per_N_pg2

    def _diags(self):
        indices = np.arange(self.dim)[:, np.newaxis]
        bin_reps = (indices >> np.arange(self.N)[::-1]) & 1
        spins = 1 - 2 * bin_reps
        spins_prime = np.hstack( (spins[:, 1:], spins[:, 0:1]) )
        self.diag_elements = - (spins * spins_prime).sum(axis=1)
        self.diag_elements = torch.from_numpy(self.diag_elements).to(self.device, 
                dtype=torch.float64)

    def _flips_basis(self):
        masks = torch.Tensor([1 << i for i in range(self.N)]).to(self.device).long()
        basis = torch.arange(self.dim).to(self.device)[:, None]
        self.flips_basis = basis ^ masks

    def setpHpg(self):
        self.pHpgmatrix = torch.zeros(self.dim, self.dim).to(self.device, 
                dtype=torch.float64)
        self.pHpgmatrix[self.flips_basis.T, torch.arange(self.dim).to(self.device)] = - 1.0

    def pHpg(self, v):
        """
            The 1st derivative of the Hamiltonian operator H of the model.
        pHpg = \\frac{\partial H}{\partial g}
             = - \sum_{i=0}{N-1} \sigma_i^x
        """
        resultv = - v[self.flips_basis].sum(dim=1)
        return resultv

    def setHmatrix(self):
        """
            Set the Hamiltonian of the model, which is a (Hermitian) square matrix
        represented as a normal torch Tensor.
            The resulting Hamiltonian matrix is stored in `self.Hmatrix`.

        Note: The applicability of this method is limited by Lattice size N. 
            To construct the Hamiltonian for larger N(~> 10, say), use
            the method `H` below.
        """
        diagmatrix = torch.diag(self.diag_elements)
        offdiagmatrix = torch.zeros(self.dim, self.dim).to(self.device, 
                dtype=torch.float64)
        offdiagmatrix[self.flips_basis.T, torch.arange(self.dim).to(self.device)] = - self.g

        # Introduce a small random noise in the Hamiltonian to avoid devide-by-zero
        #   problem when calculating 2nd derivative of E0 using AD of torch.
        randommatrix = 1e-12 * torch.randn(self.dim, self.dim).to(self.device, 
                dtype=torch.float64)
        randommatrix = 0.5 * (randommatrix + randommatrix.T)
        #randommatrix = torch.zeros(self.dim, self.dim).to(torch.float64)

        self.Hmatrix = diagmatrix + offdiagmatrix + randommatrix

    def H(self, v):
        """
            The Hamiltonian of the model, which is a "sparse" linear tranformation
        that takes a vector as input and returns another vector as output.
        """
        resultv = v * self.diag_elements \
                  - self.g * v[self.flips_basis].sum(dim=1)
        return resultv

    def Hadjoint_to_gadjoint(self, v1, v2):
        return self.pHpg(v2).matmul(v1)[None]
