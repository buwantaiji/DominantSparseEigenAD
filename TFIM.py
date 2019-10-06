import numpy as np
import torch
from Lanczos_torch import symeigLanczos

class TFIM(object):
    """
        Direct diagonalization solver of 1D Transverse Field Ising Model(TFIM).
        H = - \sum_{i=0}^{N-1} (g\sigma_i^x + \sigma_i^z \sigma_{i+1}^z)
    """
    def __init__(self, N):
        self.N = N
        self.dim = 2**N
        print("Lattice size N = %d" % self.N)
        self._diags()
        self._flips_basis()
        print("Model initialization completed.")

    def analytic_result(self):
        """
            Analytic result of the ground state energy, based on Jordan-Wigner transformation.
            The formula may be a little problematic for finite lattice size N.
        """
        ks = np.linspace(-(self.N-1)/2, (self.N-1)/2, num=N) / self.N * 2 * np.pi
        epsilon_ks = 2 * np.sqrt(self.g**2 - 2 * self.g * np.cos(ks) + 1)
        self.E0_per_N = - 0.5 * epsilon_ks.sum() / self.N

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

    def H(self, v):
        """
            The Hamiltonian of the model, which is a "sparse" linear tranformation
            that takes a vector as input and returns another vector as output.
        """
        resultv = v * self.diag_elements \
                  - self.g * v[self.flips_basis].sum(dim=1)
        return resultv

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 14
    model = TFIM(N)
    k = 300
    Npoints = 50
    gs = np.linspace(0.0, 3.0, num=Npoints)
    E0s_per_N_computation = np.empty(Npoints)
    E0s_per_N_analytic = np.empty(Npoints)
    for i in range(Npoints):
        model.g = gs[i]
        model.analytic_result()
        E0s_per_N_analytic[i] = model.E0_per_N
        E0, _ = symeigLanczos(model.H, k, extreme="min", sparse=True, dim=model.dim)
        E0s_per_N_computation[i] = E0.item() / model.N
        print("g = ", gs[i], "     Analytic result: ", E0s_per_N_analytic[i], 
                "     Direct diagonalization result: ", E0s_per_N_computation[i])
    plt.plot(gs, E0s_per_N_analytic, label="Analytic result")
    plt.plot(gs, E0s_per_N_computation, label="Direct diagonalization")
    plt.legend()
    plt.xlabel("$g$")
    plt.ylabel("$\\frac{E_0}{N}$")
    plt.title("Ground state energy per site of 1D TFIM\n" \
            "$H = - \\sum_{i=0}^{N-1} (g\\sigma_i^x + \\sigma_i^z \\sigma_{i+1}^z)$\n" \
            "$N=%d$" % model.N)
    plt.show()
