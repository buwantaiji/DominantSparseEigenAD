"""
    Plot the typical results of the TFIM example.
"""
import numpy as np
import matplotlib.pyplot as plt

for N in [10, 16, 20]:
    data_E0 = np.load("datas/E0_N_%d.npz" % N)
    gs = data_E0["gs"]
    #E0s = data_E0["E0s"]
    #dE0s = data_E0["dE0s"]
    d2E0s = data_E0["d2E0s"]
    plt.plot(gs, d2E0s, label="$N$ = %d" % N)
plt.legend()
plt.xlabel("$g$")
plt.ylabel("$\\frac{1}{N} \\frac{\\partial^2 E_0}{\\partial g^2}$")
plt.savefig("datas/d2E0s.jpg")
plt.show()

for N in [10, 16, 20]:
    data_chiF = np.load("datas/chiF_N_%d.npz" % N)
    gs = data_chiF["gs"]
    chiFs = data_chiF["chiFs"]
    plt.plot(gs, chiFs, label="$N$ = %d" % N)
plt.legend()
plt.xlabel("$g$")
plt.ylabel("$\\chi_F$")
plt.savefig("datas/chiFs.jpg")
plt.show()
