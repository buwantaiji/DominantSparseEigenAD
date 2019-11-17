"""
    Computation to machine precision of the ground state energy (per site)
of 1D Transverse Field Ising Model (TFIM) for various values of the parameter g.
"""
import numpy as np
import scipy.integrate as integrate

def E0_sum(N, g):
    ks = np.linspace(-(N-1)/2, (N-1)/2, num=N) / N * 2 * np.pi
    epsilon_ks = 2 * np.sqrt(g**2 - 2 * g * np.cos(ks) + 1)
    E0 = - 0.5 * epsilon_ks.sum() / N
    return E0

def E0_integrate(g):
    f = lambda k: -1. / (2. * np.pi) * np.sqrt(g**2 - 2 * g * np.cos(k) + 1)
    E0, error = integrate.quadrature(f, -np.pi, np.pi, tol=1e-16, rtol=1e-16, maxiter=2000)
    return E0, error

if __name__ == "__main__":
    gs = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20])
    E0s_sum = np.empty(gs.size)
    for i in range(gs.size):
        g = gs[i]
        print("g =", g)
        upper = 10 if g == 1.00 else 8
        for exponent in range(2, upper):
            E0 = E0_sum(10 ** exponent, g)
            print("N = 10^%d, \t\tE0 = %.15f" % (exponent, E0))
            if exponent == upper - 1:
                E0s_sum[i] = E0
        #print("E0_integrate = %.15f" % E0_integrate(g)[0])

    filename = "datas/E0_sum.npz"
    np.savez(filename, gs=gs, E0s=E0s_sum)
