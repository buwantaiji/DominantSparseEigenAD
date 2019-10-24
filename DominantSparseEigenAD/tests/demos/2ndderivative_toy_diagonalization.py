import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_dpsi0(a, b, g):
    squareroot = np.sqrt((a - b)**2 + 4 * g**2)
    x1 = -g
    x2 = 0.5 * (a - b + squareroot)
    dx1 = -1
    dx2 = 2 * g / squareroot
    module = np.sqrt(0.5 * (squareroot**2 + (a - b) * squareroot))
    dmodule = 1 / (4 * module) * (8 * g + 4 * g * (a - b) / squareroot)
    return (dx1 * module - x1 * dmodule) / module**2, \
           (dx2 * module - x2 * dmodule) / module**2


a = 2.5; b = 4.7
diag = torch.Tensor([[a, 0.0], [0.0, b]]).to(torch.float64)
offdiag = torch.Tensor([[0.0, 1.0], [1.0, 0.0]]).to(torch.float64)

Npoints = 50
ps = np.linspace(1e-5, 4.0, num=Npoints)
E0s = np.empty(Npoints)
dE0s = np.empty(Npoints)
d2E0s = np.empty(Npoints)
E0s_computation = np.empty(Npoints)
dE0s_computation = np.empty(Npoints)
d2E0s_computation = np.empty(Npoints)

for i in range(Npoints):
    g = ps[i]
    squareroot = np.sqrt((a - b)**2 + 4 * g**2)
    E0s[i] = 0.5 * (a + b - squareroot)
    dE0s[i] = - 2 * g / squareroot
    d2E0s[i] = - 2 * (a - b)**2 / squareroot**3

    dpsi0 = np.array([compute_dpsi0(a, b, g)])

    g = torch.Tensor([g]).to(torch.float64)
    g.requires_grad_(True)
    H = diag + g * offdiag
    Es, psis = torch.symeig(H, eigenvectors=True)

    psi0 = psis[:, 0]
    dpsi0_computation = torch.empty(2, dtype=torch.float64)
    I = torch.eye(2, dtype=torch.float64)
    for idx in range(2):
        dpsi0_computation[idx], = torch.autograd.grad(psi0, g, grad_outputs=I[:, idx], retain_graph=True)
    #dpsi0_computation, = torch.autograd.grad(psi0, g, grad_outputs=I, retain_graph=True)

    E0 = Es[0]
    dE0, = torch.autograd.grad(E0, g, create_graph=True)
    d2E0, = torch.autograd.grad(dE0, g)
    E0s_computation[i] = E0.item()
    dE0s_computation[i] = dE0.item()
    d2E0s_computation[i] = d2E0.item()

    print(ps[i], E0s[i], E0s_computation[i], 
          dE0s[i], dE0s_computation[i], 
          d2E0s[i], d2E0s_computation[i], 
          dpsi0, dpsi0_computation.numpy())
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
