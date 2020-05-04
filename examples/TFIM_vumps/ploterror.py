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

vumps_data_dir = "datas/E0s_general1/"
errors = np.empty((gs.size, Ds.size))
for col in range(Ds.size):
    E0_D = np.load(vumps_data_dir + "D_%d.npz" % Ds[col])["E0s"]
    #error_D = np.log10( np.abs((E0_D - E0s) / E0s) )
    error_D = np.abs((E0_D - E0s) / E0s)
    errors[:, col] = error_D

#for row in range(gs.size):
for i in range(1, 5):
    for row in [4 - i, 4, 4 + i]:
        plt.plot(Ds, errors[row, :], "o-", label=r"$g = %.2f$" % gs[row])
    plt.legend()
    plt.xlabel(r"$D$")
    plt.ylabel(r"Energy relative error")
    plt.yscale("log")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(vumps_data_dir + "error%d.pdf" % i)
    plt.show()
