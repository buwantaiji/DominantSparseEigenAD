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

    def analytic_results(self):
        """
            Some analytic results of the model, based on Jordan-Wigner transformation.
        The formulas may be a little problematic for finite lattice size N.

        self.E0_per_N:      E0 / N
        self.pE0_per_N_pg:  \partial (E0 / N) / \partial g
        self.p2E0_per_N_pg2: \partial2 (E0 / N) / \partial g2
        """
        ks = np.linspace(-(self.N-1)/2, (self.N-1)/2, num=N) / self.N * 2 * np.pi
        epsilon_ks = 2 * np.sqrt(self.g**2 - 2 * self.g * np.cos(ks) + 1)
        self.E0_per_N = - 0.5 * epsilon_ks.sum() / self.N
        pepsilon_ks_pg = 4 * (self.g - np.cos(ks)) / epsilon_ks
        self.pE0_per_N_pg = - 0.5 * pepsilon_ks_pg.sum() / self.N
        p2epsilon_ks_pg2 = 16 * np.sin(ks)**2 / epsilon_ks**3
        self.p2E0_per_N_pg2 = - 0.5 * p2epsilon_ks_pg2.sum() / self.N

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

    def pHpg(self, v):
        """
            The 1st derivative of the Hamiltonian operator H of the model.
        pHpg = \\frac{\partial H}{\partial g}
             = - \sum_{i=0}{N-1} \sigma_i^x
        """
        resultv = - v[self.flips_basis].sum(dim=1)
        return resultv

    def H(self, v):
        """
            The Hamiltonian of the model, which is a "sparse" linear tranformation
        that takes a vector as input and returns another vector as output.
        """
        resultv = v * self.diag_elements \
                  - self.g * v[self.flips_basis].sum(dim=1)
        return resultv

if __name__ == "__main__":
    from CG_torch import CG
    import matplotlib.pyplot as plt
    N = 18
    model = TFIM(N)
    k = 300
    Npoints = 100
    gs = np.linspace(0.0, 2.0, num=Npoints)
    E0s_per_N_computation = np.empty(Npoints)
    E0s_per_N_analytic = np.empty(Npoints)
    pE0s_per_N_pg_computation = np.empty(Npoints)
    pE0s_per_N_pg_analytic = np.empty(Npoints)
    p2E0s_per_N_pg2_computation = np.empty(Npoints)
    p2E0s_per_N_pg2_analytic = np.empty(Npoints)
    for i in range(Npoints):
        model.g = gs[i]

        model.analytic_results()
        E0s_per_N_analytic[i] = model.E0_per_N
        pE0s_per_N_pg_analytic[i] = model.pE0_per_N_pg
        p2E0s_per_N_pg2_analytic[i] = model.p2E0_per_N_pg2

        E0, psi0 = symeigLanczos(model.H, k, extreme="min", sparse=True, dim=model.dim)
        E0s_per_N_computation[i] = E0.item() / model.N
        pE0s_per_N_pg_computation[i] = model.pHpg(psi0).matmul(psi0).item() / model.N
        A = lambda v: model.H(v) - E0 * v
        b = 2 * model.pHpg(psi0)
        b = b - torch.matmul(psi0, b) * psi0
        initialx = torch.randn(model.dim, dtype=b.dtype)
        initialx = initialx - torch.matmul(psi0, initialx) * psi0
        lambda0 = CG(A, b, initialx, sparse=True)
        p2E0s_per_N_pg2_computation[i] = - model.pHpg(psi0).matmul(lambda0) / model.N
        
        print("g = ", gs[i],
                "   E0_analytic: ", E0s_per_N_analytic[i], 
                "   E0_computation: ", E0s_per_N_computation[i],
                "   pE0pg_analytic: ", pE0s_per_N_pg_analytic[i],
                "   pE0pg_computation: ", pE0s_per_N_pg_computation[i],
                "   p2E0pg2_analytic: ", p2E0s_per_N_pg2_analytic[i],
                "   p2E0pg2_computation: ", p2E0s_per_N_pg2_computation[i])
    plt.plot(gs, E0s_per_N_analytic, label="Analytic result")
    plt.plot(gs, E0s_per_N_computation, label="Direct diagonalization")
    plt.legend()
    plt.xlabel("$g$")
    plt.ylabel("$\\frac{E_0}{N}$")
    plt.title("Ground state energy per site of 1D TFIM\n" \
            "$H = - \\sum_{i=0}^{N-1} (g\\sigma_i^x + \\sigma_i^z \\sigma_{i+1}^z)$\n" \
            "$N=%d$" % model.N)
    plt.show()
    plt.plot(gs, pE0s_per_N_pg_analytic, label="Analytic result")
    plt.plot(gs, pE0s_per_N_pg_computation, label="Direct diagonalization")
    plt.legend()
    plt.xlabel("$g$")
    plt.ylabel("$\\frac{1}{N} \\frac{\\partial E_0}{\\partial g}$")
    plt.title("1st derivative w.r.t. $g$ of ground state energy per site of 1D TFIM\n" \
            "$H = - \\sum_{i=0}^{N-1} (g\\sigma_i^x + \\sigma_i^z \\sigma_{i+1}^z)$\n" \
            "$N=%d$" % model.N)
    plt.show()
    plt.plot(gs, p2E0s_per_N_pg2_analytic, label="Analytic result")
    plt.plot(gs, p2E0s_per_N_pg2_computation, label="Direct diagonalization")
    plt.legend()
    plt.xlabel("$g$")
    plt.ylabel("$\\frac{1}{N} \\frac{\\partial^2 E_0}{\\partial g^2}$")
    plt.title("2nd derivative w.r.t. $g$ of ground state energy per site of 1D TFIM\n" \
            "$H = - \\sum_{i=0}^{N-1} (g\\sigma_i^x + \\sigma_i^z \\sigma_{i+1}^z)$\n" \
            "$N=%d$" % model.N)
    plt.show()
