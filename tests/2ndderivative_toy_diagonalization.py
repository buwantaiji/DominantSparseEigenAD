import torch
import numpy as np
import matplotlib.pyplot as plt

a = 9.0; b = 4.7
diag = torch.Tensor([[a, 0.0], [0.0, b]]).to(torch.float64)
offdiag = torch.Tensor([[0.0, 1.0], [1.0, 0.0]]).to(torch.float64)

Npoints = 50
ps = np.linspace(0.0, 4.0, num=Npoints)
E0s = np.empty(Npoints)
dE0s = np.empty(Npoints)
d2E0s = np.empty(Npoints)
E0s_computation = np.empty(Npoints)
dE0s_computation = np.empty(Npoints)
d2E0s_computation = np.empty(Npoints)

for i in range(Npoints):
    p = ps[i]
    squareroot = np.sqrt((a - b)**2 + 4 * p**2)
    E0s[i] = 0.5 * (a + b - squareroot)
    dE0s[i] = - 2 * p / squareroot
    d2E0s[i] = - 2 * (a - b)**2 / squareroot**3

    g = torch.Tensor([p]).to(torch.float64)
    g.requires_grad_(True)
    H = diag + g * offdiag
    Es, psis = torch.symeig(H, eigenvectors=True)
    E0 = Es[0]
    E0s_computation[i] = E0.item()
    dE0, = torch.autograd.grad(E0, g, create_graph=True)
    dE0s_computation[i] = dE0.item()
    d2E0, = torch.autograd.grad(dE0, g)
    d2E0s_computation[i] = d2E0.item()
    print(p, E0s[i], E0s_computation[i], 
          dE0s[i], dE0s_computation[i], 
          d2E0s[i], d2E0s_computation[i])
plt.plot(ps, E0s, label="Analytic result")
plt.plot(ps, E0s_computation, label="Direct diagonalization")
plt.legend()
plt.xlabel("$g$")
plt.ylabel("$E_0$")
plt.title("0th derivative")

plt.show()
plt.plot(ps, dE0s, label="Analytic result")
plt.plot(ps, dE0s_computation, label="AD")
plt.legend()
plt.xlabel("$g$")
plt.ylabel("$\\frac{\\partial E_0}{\\partial g}$")
plt.title("1st derivative")

plt.show()
plt.plot(ps, d2E0s, label="Analytic result")
plt.plot(ps, d2E0s_computation, label="AD")
plt.legend()
plt.xlabel("$g$")
plt.ylabel("$\\frac{\\partial^2 E_0}{\\partial g^2}$")
plt.title("2nd derivative")

plt.show()
