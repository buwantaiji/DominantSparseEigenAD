"""
    A tiny example demonstrating the mechanism of performing higher order derivatives
of a computation process in Pytorch. The running time scales exponentially with
the order of derivatives, which reveals the exponentially expanding of corresponding
computation graph involved.
"""
import torch
import scipy.special as sp

class Bessel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, n, x):
        ctx.n = n
        ctx.save_for_backward(x)
        return torch.as_tensor(sp.iv(n, x.detach().cpu().numpy()), device=x.device)

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        #cf http://functions.wolfram.com/Bessel-TypeFunctions/BesselI/20/01/02/0003/
        return None, 0.5* grad_out *(Bessel.apply(ctx.n - 1, x)+Bessel.apply(ctx.n + 1, x))

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import time

    n  = 4
    x = torch.linspace(-7, 7, 100, requires_grad=True)
    bessel = Bessel.apply

    for i in range(18):
        start = time.time()
        if (i==0):
            y = bessel(n, x)
        else:
            y, = torch.autograd.grad(y, x, grad_outputs=torch.ones(y.shape[0]), create_graph=True)
        end = time.time()
        print("The %dth derivative: %f" % (i, end - start))
        plt.plot(x.detach().numpy(), y.detach().numpy(), '-', label='$%g$'%(i))

    plt.legend()
    plt.show()
