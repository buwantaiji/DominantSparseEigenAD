"""
    Plot errors of the ground state energy of TFIM calculated through
variational optimization of MPS, relative to the analytical result.
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

Ds = np.arange(5, 100 + 1, step=5)

analytic_E0 = np.load("datas/E0_sum.npz")
gs = analytic_E0["gs"]
E0s = analytic_E0["E0s"]

gs, E0s = gs[3:6], E0s[3:6]

vumps_data_dir = "datas/E0s_general/"
errors = np.empty((gs.size, Ds.size))
for row in range(gs.size):
    E0_g = np.load(vumps_data_dir + "g_%.2f.npz" % gs[row])["E0s"]
    print("g:", gs[row], "E0_analytic:", E0s[row], "vumps:", E0_g)
    error_g = np.abs((E0_g - E0s[row]) / E0s[row])
    errors[row, :] = error_g

for row in range(gs.size):
    plt.plot(Ds, errors[row, :], "o-", label=r"$g = %.2f$" % gs[row])
plt.legend()
plt.xlabel(r"$D$")
plt.ylabel(r"Energy relative error")
plt.yscale("log")
plt.subplots_adjust(bottom=0.15, left=0.15)
# plt.savefig(vumps_data_dir + "error.pdf")
plt.show()
