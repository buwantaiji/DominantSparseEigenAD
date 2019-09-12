import torch
from Lanczos_torch import symeigLanczos

class Dimer2D(torch.nn.Module):
    def __init__(self, D):
        super(Dimer2D, self).__init__()
        self.theoretical_value = 0.2915609040308188
        d = 2
        self.dim_Wang = d * D**2
        self.dim_Gong = D**2
        self.T = torch.zeros(d, d, d, d)
        self.T[0, 0, 0, 1] = 1.0
        self.T[0, 0, 1, 0] = 1.0
        self.T[0, 1, 0, 0] = 1.0
        self.T[1, 0, 0, 0] = 1.0
        A = torch.randn(d, D, D)
        A = A + A.permute(0, 2, 1)
        self.A = torch.nn.Parameter(A)

    def forward(self):
        A = 0.5 * (self.A + self.A.permute(0, 2, 1))
        Wang = torch.einsum("okl,pmn,opij->kimljn", A, A, self.T).reshape(self.dim_Wang, self.dim_Wang)
        Gong = torch.einsum("okl,omn->kmln", A, A).reshape(self.dim_Gong, self.dim_Gong)
        #print("Wang.size", Wang.shape)
        #print("Gong.size", Gong.shape)
        k = 100
        with torch.no_grad():
            _, eigvector_max_Wang = symeigLanczos(Wang, k, extreme="max")
            _, eigvector_max_Gong = symeigLanczos(Gong, k, extreme="max")
        lambda_max_Wang = eigvector_max_Wang.matmul(Wang).matmul(eigvector_max_Wang)
        lambda_max_Gong = eigvector_max_Gong.matmul(Gong).matmul(eigvector_max_Gong)
        result = torch.log(lambda_max_Wang / lambda_max_Gong)
        """
        lambdas_Wang, _ = torch.symeig(Wang, eigenvectors=True)
        lambdas_Gong, _ = torch.symeig(Gong, eigenvectors=True)
        result = torch.log(lambdas_Wang[-1] / lambdas_Gong[-1])
        """
        return result

D = 16
model = Dimer2D(D)
optimizer = torch.optim.LBFGS(model.parameters(), max_iter=20, tolerance_grad=1E-7)
def closure():
    result = - model()
    optimizer.zero_grad()
    result.backward()
    return result

import time
iter_num = 100
start_all = time.time()
for i in range(iter_num):
    start = time.time()
    result = optimizer.step(closure)
    end = time.time()
    print("iter: ", i, -result.item(),
            (-result - model.theoretical_value).item() / model.theoretical_value,
            "time: ", end - start)
end_all = time.time()
print("Total time: ", end_all - start_all)
