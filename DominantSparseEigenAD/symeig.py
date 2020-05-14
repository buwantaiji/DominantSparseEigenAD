import torch
from .Lanczos import symeigLanczos

class DominantSymeig(torch.autograd.Function):
    """
        Function primitive of dominant real symmetric eigensolver, where the matrix
    is represented in normal form as a torch.Tensor.
    
    input: A -- the real symmetric matrix A.
           k -- number of Lanczos vectors requested.(doesn't need gradient)
    output: eigval -- the smallest eigenvalue of A.
            eigvector -- corresponding (non-degenerate) eigenvector.
    """ 
    @staticmethod
    def forward(ctx, A, k, device=torch.device("cpu")):
        eigval, eigvector = symeigLanczos(A, k, device=device, extreme="min")
        ctx.save_for_backward(A, eigval, eigvector)
        ctx.device = device
        return eigval, eigvector
    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvector):
        from .CG import CGSubspace
        A, eigval, eigvector = ctx.saved_tensors
        device = ctx.device
        Aprime = A - eigval * torch.eye(A.shape[0], device=device, dtype=A.dtype)
        cg = CGSubspace.apply
        b = grad_eigvector - torch.matmul(eigvector, grad_eigvector) * eigvector
        lambda0 = cg(Aprime, b, eigvector)
        grad_A = (grad_eigval * eigvector - lambda0)[:, None] * eigvector
        grad_k = grad_device = None
        return grad_A, grad_k, grad_device

def setDominantSparseSymeig(A, Aadjoint_to_gadjoint):
    """
        Function primitive of dominant real symmetric eigensolver of a "sparse" matrix
    represented as a function.

        As a workaround of the fact that Pytorch doesn't support taking gradient 
    of objects of type other than torch.Tensor, the computation graph of this primitive 
    is slightly wrapped compared to DominantSymeig, which is the primitive of the case
    of the normal torch.Tensor representation, and is schematically shown below.
        ---------
        |     --|--> eigval
        |    /  |
    g --|-->A   |
        |    \  |
        |     --|--> eigvector
        ---------
    input: g -- The parameter(s) of interest of the matrix A, whose gradients are requested.
                In current version, g must be a torch.Tensor of arbitrary shape.
           k -- number of Lanczos vectors requested. (doesn't need gradient)
           dim -- The dimension of the square matrix A. (doesn't need gradient)
    output: eigval -- the smallest eigenvalue of A.
            eigvector -- corresponding (non-degenerate) eigenvector.
    
    USER NOTE: The mechanism of wrapping relies on user's providing two quantities:
        A -- The "sparse" representation of the matrix A as a function.
        Aadjoint_to_gadjoint -- A function that receive the adjoint of the matrix A
            as input, and return the adjoint of the pamameters(g) as output.

            The input should be of the form of two vectors represented as torch.Tensor, 
        say, v1 and v2, and the adjoint of A = v1 * v2^T.(outer product)
            User may do whatever he want to get the adjoint of g using these
        two vectors.
    """
    global DominantSparseSymeig 
    from .CG import setCGSubspaceSparse
    setCGSubspaceSparse(A, Aadjoint_to_gadjoint)
    from .CG import CGSubspaceSparse
    @staticmethod
    def forward(ctx, g, k, dim, device=torch.device("cpu")):
        eigval, eigvector = symeigLanczos(A, k, device=device, 
                extreme="min", sparse=True, dim=dim)
        ctx.save_for_backward(g, eigval, eigvector)
        return eigval, eigvector
    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvector):
        cg = CGSubspaceSparse.apply
        g, eigval, eigvector = ctx.saved_tensors
        b = grad_eigvector - torch.matmul(eigvector, grad_eigvector) * eigvector
        lambda0 = cg(g, eigval, b, eigvector)
        grad_A = grad_eigval * eigvector - lambda0, eigvector
        v1, v2 = grad_A
        grad_g = Aadjoint_to_gadjoint(v1, v2)
        grad_k = grad_dim = grad_device = None
        return grad_g, grad_k, grad_dim, grad_device
    DominantSparseSymeig = type("DominantSparseSymeig", (torch.autograd.Function, ), 
            {"forward": forward, "backward": backward})

