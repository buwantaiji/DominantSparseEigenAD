"""
    Plot typical results of the TFIM example.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.serif'] = 'Computer Modern Roman'

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.fontsize'] = 16

#colors = ["C7", "C8", "C3"]
colors = ["red", "green", "blue"]

# Plot 0th, 1st, 2nd derivatives of the ground state energy per site E0.
Ns = [10, 16, 20]
for i, N in enumerate(Ns):
    data_E0 = np.load("datas/E0_N_%d.npz" % N)
    gs = data_E0["gs"]
    #E0s = data_E0["E0s"]
    #dE0s = data_E0["dE0s"]
    d2E0s = data_E0["d2E0s"]
    plt.plot(gs, d2E0s, label="$N = %d$" % N, color=colors[i])
plt.legend()
plt.xlabel(r"$g$")
plt.ylabel(r"$\frac{\partial^2 E_0}{\partial g^2}$")
plt.subplots_adjust(bottom=0.15, left=0.2)
# plt.savefig("datas/d2E0s.pdf")
plt.show()

# Plot fidelity susceptibility chiF.
for i, N in enumerate(Ns):
    data_chiF = np.load("datas/chiF_N_%d.npz" % N)
    gs = data_chiF["gs"]
    chiFs = data_chiF["chiFs"]
    plt.plot(gs, chiFs, label="$N = %d$" % N, color=colors[i])
plt.legend()
plt.xlabel(r"$g$")
plt.ylabel(r"$\chi_F$")
plt.subplots_adjust(bottom=0.15)
# plt.savefig("datas/chiFs.pdf")
plt.show()
