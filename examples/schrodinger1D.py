import numpy as np
import torch
import DominantSparseEigenAD.symeig as symeig
import matplotlib.pyplot as plt

class Schrodinger1D(torch.nn.Module):
    def __init__(self, xmin, xmax, N, xmesh):
        super(Schrodinger1D, self).__init__()
        self.xmesh = xmesh
        self.N = N
        self.h = (xmax - xmin) / N
        self.K = -0.5/self.h**2 * \
                    (torch.diag(-2 * torch.ones(N, dtype=xmesh.dtype))
                    + torch.diag(torch.ones(N - 1, dtype=xmesh.dtype), diagonal=1)
                    + torch.diag(torch.ones(N - 1, dtype=xmesh.dtype), diagonal=-1))
        self.potential = torch.nn.Parameter(0.5 * xmesh**2)

    def Hsparse(self, v):
        """
            The Hamiltonian, which is a sparse matrix represented as a linear
        function that takes a vector as input, and returns another vector as output.
        """
        zero = torch.zeros(1, dtype=torch.float64)
        v1 = torch.cat((v[1:], zero))
        v2 = torch.cat((zero, v[:-1]))
        return -0.5/self.h**2 * (-2 * v + v1 + v2) \
                + self.potential * v

    def Hadjoint_to_padjoint(self, v1, v2):
        """
            Another function that have to be supplied to make the DominantSparseEigenAD
        Function primitive work properly.
        """
        return v1 * v2
    """
        Below are three approaches to perform the forward computation of the 
    optimization process.
        In current problem, the `target` plays the same role as the input 
    in a neural network. Given a target, the parameters of the model -- the potential
    in current case -- are optimized.
    """
    def forward_torch(self, target):
        """
            Full-diagonalization of Hamiltonian using the "symeig" function
        built in Pytorch.
        """
        V = torch.diag(self.potential)
        H = self.K + V
        _, psis = torch.symeig(H, eigenvectors=True)
        self.psi0 = psis[:, 0]
        loss = 1. - (self.psi0.abs() * target).sum()
        return loss
    def forward_matrixAD(self, target, k):
        """
            Dominant diagonalization of Hamiltonian using DominantSymeig primitive, 
        where the Hamiltonian is represented in normal form as a torch.Tensor.
        """
        dominant_symeig = symeig.DominantSymeig.apply
        V = torch.diag(self.potential)
        H = self.K + V
        _, self.psi0 = dominant_symeig(H, k)
        loss = 1. - (self.psi0.abs() * target).sum()
        return loss
    def forward_sparseAD(self, target, k):
        """
            Dominant diagonalization of Hamiltonian using DominantSparseSymeig
        primitive, where the Hamiltonian is represented in sparse form as a function.
        """
        symeig.setDominantSparseSymeig(self.Hsparse, self.Hadjoint_to_padjoint)
        dominant_sparse_symeig = symeig.DominantSparseSymeig.apply
        _, self.psi0 = dominant_sparse_symeig(self.potential, k, self.N)
        loss = 1. - (self.psi0.abs() * target).sum()
        return loss

    def plot(self, target):
        plt.cla()
        plt.plot(self.xmesh, target, label="target")
        plt.plot(self.xmesh, self.psi0.detach().abs().numpy(), 
                label="$\\psi_0$ corresponding to current potential $V$")
        plt.plot(self.xmesh, self.potential.detach().numpy() / 20000, label="$V$")
        plt.xlabel("$x$")
        plt.legend()
        plt.draw()
       
if __name__ == "__main__":
    # Set axis and the target wave function.
    xmin, xmax, N = -1., 1., 300
    xmesh = np.linspace(xmin, xmax, num=N, endpoint=False)
    k = 300

    target = np.zeros(N)
    idx = (np.abs(xmesh) < 0.5)
    target[idx] = 1. - np.abs(xmesh[idx])
    target /= np.linalg.norm(target)


    xmesh = torch.from_numpy(xmesh).to(torch.float64)
    target = torch.from_numpy(target).to(torch.float64)

    model = Schrodinger1D(xmin, xmax, N, xmesh)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=10, 
            tolerance_change = 1E-7, tolerance_grad=1E-7, line_search_fn='strong_wolfe')

    def closure():
        import time
        optimizer.zero_grad()

        start1 = time.time()
        # Feel free to check out different approaches of matrix diagonalization here!
        #loss = model.forward_torch(target)
        #loss = model.forward_matrixAD(target, k)
        loss = model.forward_sparseAD(target, k)
        end1 = time.time()

        start2 = time.time()
        loss.backward()
        end2 = time.time()
        print("forward time: ", end1 - start1, 
                "backward time: ", end2 - start2)
        return loss

    plt.ion()
    for i in range(50):
        loss = optimizer.step(closure)
        print(i, loss.item())
        model.plot(target)
        plt.pause(0.01)
    plt.ioff()

    model.plot(target)
    plt.show()
