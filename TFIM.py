import numpy as np
import torch

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
        self.diag_elements = torch.from_numpy(self.diag_elements).to(torch.float64)

    def _flips_basis(self):
        masks = torch.Tensor([1 << i for i in range(self.N)]).long()
        basis = torch.arange(self.dim)[:, None]
        self.flips_basis = basis ^ masks

    def setpHpg(self):
        self.pHpgmatrix = torch.zeros(self.dim, self.dim).to(torch.float64)
        self.pHpgmatrix[self.flips_basis.T, torch.arange(self.dim)] = - 1.0

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
        offdiagmatrix = torch.zeros(self.dim, self.dim).to(torch.float64)
        offdiagmatrix[self.flips_basis.T, torch.arange(self.dim)] = - self.g

        # Introduce a small random noise in the Hamiltonian to avoid devide-by-zero
        #   problem when calculating 2nd derivative of E0 using AD of torch.
        randommatrix = 1e-12 * torch.randn(self.dim, self.dim).to(torch.float64)
        randommatrix = 0.5 * (randommatrix + randommatrix.T)
        #randommatrix = torch.zeros(self.dim, self.dim).to(torch.float64)

        self.Hmatrix = diagmatrix + offdiagmatrix + randommatrix

    def E0_derivatives_AD_torch(self):
        Es, psis = torch.symeig(self.Hmatrix, eigenvectors=True)
        E0 = Es[0]
        dE0, = torch.autograd.grad(E0, model.g, create_graph=True)
        d2E0, = torch.autograd.grad(dE0, model.g)
        return E0.detach().item() / model.N, \
               dE0.detach().item() / model.N, \
               d2E0.detach().item() / model.N

    def H(self, v):
        """
            The Hamiltonian of the model, which is a "sparse" linear tranformation
        that takes a vector as input and returns another vector as output.
        """
        resultv = v * self.diag_elements \
                  - self.g.detach() * v[self.flips_basis].sum(dim=1)
        return resultv

    def E0_derivatives_AD_Lanczos(self, k):
        E0, psi0 = symeigLanczos(self.H, k, extreme="min", sparse=True, dim=self.dim)
        dE0 = self.pHpg(psi0).matmul(psi0) 

        A = lambda v: self.H(v) - E0 * v
        b = 2 * self.pHpg(psi0)
        b = b - torch.matmul(psi0, b) * psi0
        initialx = torch.randn(self.dim, dtype=b.dtype)
        initialx = initialx - torch.matmul(psi0, initialx) * psi0
        lambda0 = CG(A, b, initialx, sparse=True)
        d2E0 = - self.pHpg(psi0).matmul(lambda0) 

        return E0.item() / self.N, \
               dE0.item() / self.N, \
               d2E0.item() / self.N

    def Hadjoint_to_gadjoint(self, v1, v2):
        return self.pHpg(v2).matmul(v1)[None]

if __name__ == "__main__":
    from Lanczos_torch import symeigLanczos
    from CG_torch import CG
    import matplotlib.pyplot as plt

    N = 10
    model = TFIM(N)
    k = 300
    Npoints = 100
    gs = np.linspace(1e-6, 2.0, num=Npoints)
    E0s_per_N_analytic = np.empty(Npoints)
    E0s_per_N_AD_torch = np.empty(Npoints)
    E0s_per_N_AD_Lanczos = np.empty(Npoints)
    pE0s_per_N_pg_analytic = np.empty(Npoints)
    pE0s_per_N_pg_AD_torch = np.empty(Npoints)
    pE0s_per_N_pg_AD_Lanczos = np.empty(Npoints)
    p2E0s_per_N_pg2_analytic = np.empty(Npoints)
    p2E0s_per_N_pg2_AD_torch = np.empty(Npoints)
    p2E0s_per_N_pg2_AD_Lanczos = np.empty(Npoints)

    print("g    E0_analytic    E0_AD_torch    E0_AD_Lanczos    "\
          "pE0pg_analytic    pE0pg_AD_torch    pE0pg_AD_Lanczos    "\
          "p2E0pg2_analytic    p2E0pg2_AD_torch    p2E0pg2_AD_Lanczos")
    for i in range(Npoints):
        model.g = torch.Tensor([gs[i]]).to(torch.float64)
        model.g.requires_grad_(True)

        E0s_per_N_analytic[i], pE0s_per_N_pg_analytic[i], p2E0s_per_N_pg2_analytic[i] \
                = model.analytic_results()

        model.setHmatrix()
        E0s_per_N_AD_torch[i], pE0s_per_N_pg_AD_torch[i], p2E0s_per_N_pg2_AD_torch[i] \
                = model.E0_derivatives_AD_torch()

        E0s_per_N_AD_Lanczos[i], pE0s_per_N_pg_AD_Lanczos[i], p2E0s_per_N_pg2_AD_Lanczos[i] \
                = model.E0_derivatives_AD_Lanczos(k)


        print(gs[i], E0s_per_N_analytic[i], E0s_per_N_AD_torch[i], E0s_per_N_AD_Lanczos[i],
              pE0s_per_N_pg_analytic[i], pE0s_per_N_pg_AD_torch[i], pE0s_per_N_pg_AD_Lanczos[i],
              p2E0s_per_N_pg2_analytic[i], p2E0s_per_N_pg2_AD_torch[i], p2E0s_per_N_pg2_AD_Lanczos[i])

    plt.plot(gs, E0s_per_N_analytic, label="Analytic result")
    plt.plot(gs, E0s_per_N_AD_torch, label="AD: torch")
    plt.plot(gs, E0s_per_N_AD_Lanczos, label="AD: Lanczos")
    plt.legend()
    plt.xlabel("$g$")
    plt.ylabel("$\\frac{E_0}{N}$")
    plt.title("Ground state energy per site of 1D TFIM\n" \
            "$H = - \\sum_{i=0}^{N-1} (g\\sigma_i^x + \\sigma_i^z \\sigma_{i+1}^z)$\n" \
            "$N=%d$" % model.N)
    plt.show()
    plt.plot(gs, pE0s_per_N_pg_analytic, label="Analytic result")
    plt.plot(gs, pE0s_per_N_pg_AD_torch, label="AD: torch")
    plt.plot(gs, pE0s_per_N_pg_AD_Lanczos, label="AD: Lanczos")
    plt.legend()
    plt.xlabel("$g$")
    plt.ylabel("$\\frac{1}{N} \\frac{\\partial E_0}{\\partial g}$")
    plt.title("1st derivative w.r.t. $g$ of ground state energy per site of 1D TFIM\n" \
            "$H = - \\sum_{i=0}^{N-1} (g\\sigma_i^x + \\sigma_i^z \\sigma_{i+1}^z)$\n" \
            "$N=%d$" % model.N)
    plt.show()
    plt.plot(gs, p2E0s_per_N_pg2_analytic, label="Analytic result")
    plt.plot(gs, p2E0s_per_N_pg2_AD_torch, label="AD: torch")
    plt.plot(gs, p2E0s_per_N_pg2_AD_Lanczos, label="AD: Lanczos")
    plt.legend()
    plt.xlabel("$g$")
    plt.ylabel("$\\frac{1}{N} \\frac{\\partial^2 E_0}{\\partial g^2}$")
    plt.title("2nd derivative w.r.t. $g$ of ground state energy per site of 1D TFIM\n" \
            "$H = - \\sum_{i=0}^{N-1} (g\\sigma_i^x + \\sigma_i^z \\sigma_{i+1}^z)$\n" \
            "$N=%d$" % model.N)
    plt.show()
