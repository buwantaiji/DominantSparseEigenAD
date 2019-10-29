# DominantSparseEigenAD

[![Build Status](https://travis-ci.com/buwantaiji/DominantSparseEigenAD.svg?branch=master)](https://travis-ci.com/buwantaiji/DominantSparseEigenAD)

DominantSparseEigenAD is an extension of Pytorch that handles backward automatic differentiation of dominant eigen-decomposition. 



In many researches and real applications involving matrix diagonalization, typically in the context of eigenvalue problem in quantum mechanics, only a small number of eigenvalues (usually the smallest ones) and corresponding eigenvectors are of practical interest. This library provides corresponding primitives in the framework of Pytorch to automatically differentiate this process, without the need of the full spectrum of the original matrix being diagonalized. The library supports matrix representations both in "normal" form of `torch.Tensor` and in a "sparse" form of python function. Finally, it's worth noting that in principle the library can perform **arbitrarily higher order of derivatives** of the dominant eigen-decomposition process.



## Installation

```bash
$ pip install DominantSparseEigenAD
```

## Examples

Please check out the [examples](examples/) directory. 

- [Hamiltonian engineering in 1D coordinate space](examples/schrodinger1D.py). A 1D potential is fitted in order to match the desired ground-state wave function. 
- [Exact-diagonalization investigation of quantum phase transition in 1D transverse field Ising model(TFIM)](examples/TFIM/). Various orders of derivative of the ground-state energy and the fidelity susceptibility of TFIM are computed using backward AD of the dominant diagonalization process.

## Usage

The library provides two `torch.autograd.Function` primitives, `DominantSymeig` and `DominantSparseSymeig`, for the case in which the matrix to be diagonalized is represented as a normal `torch.Tensor` and as a function, respectively. To make use of a primitive, simply call the `apply` method of it. For example, the following code

```python
dominantsymeig = DominantSymeig.apply
```

will create a function named `dominantsymeig`, which can then be directly used in the computation process. 

### DominantSymeig

```python
from DominantSparseEigenAD.Lanczos import DominantSymeig

dominantsymeig = DominantSymeig.apply

# Usage
dominantsymeig(A, k)
```

The `DominantSymeig` primitive is used in the case where the matrix is represented in "normal" form. In current version, it will use the **Lanczos** algorithm to return a tuple `(eigval, eigvalue)` of the **smallest** eigenvalue and corresponding eigenvector of the matrix, both represented as `torch.Tensor`s. It accepts two arguments:

- `A` is the matrix to be diagonalized, which is represented as a `torch.Tensor`. **In current version, `A` must be real symmetric**.
- `k` is the number of Lanczos vectors requested. In typical applications, `k` is far less than the dimension N of the matrix `A`. The choice of several hundreds for `k` may be satisfactory for N up to 100000. Note that `k` should never exceeds N in any cases.

Only the gradient of the matrix `A` will be computed when performing backward AD. The argument `k` doesn't require computing gradients.

### DominantSparseSymeig

```python
import DominantSparseEigenAD.Lanczos as lanczos

lanczos.setDominantSparseSymeig(A, Aadjoint_to_padjoint)
dominantsparsesymeig = lanczos.DominantSparseSymeig.apply

# Usage
dominantsparsesymeig(p, k, dim)
```

The `DominantSparseSymeig` primitive is used in the case where the matrix is represented in "sparse" form. In current version, it will use the **Lanczos** algorithm to return a tuple `(eigval, eigvalue)` of the **smallest** eigenvalue and corresponding eigenvector of the matrix, both represented as `torch.Tensor`s. **The matrix should be considered as depending on several parameters of interest**. Here is the detailed information about the arguments:

- `p` is the parameter(s) that the matrix to be diagonalized depends on. `p` should be a `torch.Tensor`(of any shape).
- `k` is the number of Lanczos vectors requested. In typical applications, `k` is far less than the dimension N of the matrix to be diagonalized. The choice of several hundreds for `k` may be satisfactory for N up to 100000. Note that `k` should never exceeds N in any cases.
- `dim` is the dimension of the matrix to be diagonalized. 

**To make the `DominantSparseSymeig` primitive work properly, two additional quantities should be provided by users before the primitive is actually available in the running session**: (See the second line of the code above)

- `A` is the matrix to be diagonalized. **In current version, `A` must be real symmetric**. As noted above, `A` is represented in "sparse" form as a python function, which mathematically is known as a linear map that receives a vector as input and returns another vector as output. Both the input and output vectors should be `torch.Tensor`s. For example, a diagonal matrix whose diagonal elements are stored in a `torch.Tensor` `a` can be represented as a function below:

  ```python
  def diagonal(v):
      return a * v
  ```

- `Aadjoint_to_padjoint` is another python function that receives the adjoint $\overline{A}$ of the matrix `A`(i.e., gradient with respect to `A` of some scalar loss) as input and returns the adjoint $\overline{p}$ of the parameter(s) `p` as output. This function receives two vectors of equal size represented as `torch.Tensor`s, `v1` and `v2`, and computes the adjoint of `p` assuming that the adjoint of `A` can be written as
  $$
  \overline{A} = v_1 v_2^T.
  $$
  For clarity, consider two simple examples:

  1. `A` can be written as the perturbative form:
     $$
     A = A_0 + p A_1.
     $$
     where the parameter $p$ in current case is a scalar. Then we have
     $$
     \begin{align}
     \overline{p} &= \mathrm{Tr}\left(\overline{A}^T \frac{\partial A}{\partial p}\right) \\
     &= \mathrm{Tr}\left(v_2 v_1^T A_1\right) \\
     &= v_1^T A_1 v_2.
     \end{align}
     $$
     Hence, suppose the matrix $A_1$ is coded as a python function `A1`, then the function `Aadjoint_to_padjoint` can be implemented as follows:

     ```python
     def Aadjoint_to_padjoint(v1, v2):
         return A1(v2).matmul(v1)
     ```

  2. `A` can be written as
     $$
     A = A_0 + D(\mathbf{p}).
     $$
     where $D$ is a diagonal matrix whose diagonal elements correspond to the parameters $\mathbf{p}$(which in current case is a vector of size equal to the dimension of the matrix `A`). Since `A` depends on $\mathbf{p}$ only through the diagonal matrix $D$, one can similarly follow the steps above and obtains
     $$
     \overline{\mathbf{p}} = v_1 \circ v_2.
     $$
     where "$\circ$" denotes the Hadamard pairwise product. The code thus reads:

     ```python
     def Aadjoint_to_padjoint(v1, v2):
         return v1 * v2
     ```

Only the gradient of the parameter `p` will be computed when performing backward AD. The other two arguments `k` and `dim` doesn't require computing gradients.

