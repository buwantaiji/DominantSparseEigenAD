# DominantSparseEigenAD

[![Build Status](https://travis-ci.com/buwantaiji/DominantSparseEigenAD.svg?branch=master)](https://travis-ci.com/buwantaiji/DominantSparseEigenAD)

DominantSparseEigenAD is an extension of PyTorch that handles reverse-mode automatic differentiation of dominant eigen-decomposition process. 

In many researches and applications involving matrix diagonalization, typically in the context of eigenvalue problem in quantum mechanics, only a small proportion of eigenvalues (e.g., the smallest ones) and corresponding eigenvectors are of practical interest. This library provides corresponding primitives in the framework of PyTorch to automatically differentiate this process, without any direct access to the full spectrum of the matrix being diagonalized.



## Installation

```bash
$ pip install DominantSparseEigenAD
```

## Examples

Please check out the [examples](examples/) directory. 

- [Hamiltonian engineering in 1D coordinate space](examples/schrodinger1D.py). A 1D potential is fitted in order to match a given ground-state wave function. 
- [Exact-diagonalization investigation of quantum phase transition in 1D transverse field Ising model (TFIM)](examples/TFIM). [Various orders of derivative of the ground-state energy](examples/TFIM/E0.py) and the [fidelity susceptibility](examples/TFIM/chiF.py) of TFIM are computed by differentiating through the (dominant) exact diagonalization process.
- [Gradient-based optimization of the ground-state energy of TFIM using matrix product states (MPS)](examples/TFIM_vumps/). See [symmetric.py](examples/TFIM_vumps/symmetric.py) for a simple demonstration, in which the transfer matrix, whose largest-amplitude eigenvalue and corresponding eigenvector are desired, is assumed to be symmetric. For a more complete implementation allowing for any (diagonalizable) transfer matrices, see [general.py](examples/TFIM_vumps/general.py).

## Usage

The library provides several `torch.autograd.Function` primitives for typical use cases. Specifically, the matrix to be diagonalized can be either assumed to be (real) symmetric or not. It can also be represented either as a normal `torch.Tensor` or in a sparse form, such as a function or scipy `LinearOperator`. 

Typically, to make use of a primitive, simply call the `apply` method of it. For example, the following code

```python
dominantsymeig = DominantSymeig.apply
```

will create a function named `dominantsymeig` corresponding to the primitive `DominantSymeig`, which can then be directly invoked in a computation process.

### DominantSymeig

```python
from DominantSparseEigenAD.symeig import DominantSymeig

dominantsymeig = DominantSymeig.apply

# Usage
dominantsymeig(A, k)
```

This primitive is used in the case where **the matrix is assumed to be real symmetric and represented in "normal" form as a `torch.Tensor`**. In current version, it will use the Lanczos algorithm to return a tuple `(eigval, eigvector)` of the **smallest** eigenvalue and corresponding eigenvector, both represented as `torch.Tensor`s.

**Parameters**:

- **A**: `torch.Tensor` - the matrix to be diagonalized.
- **k**: `int` -  the number of Lanczos vectors requested. In typical applications, `k` is far less than the dimension $N$ of the matrix `A`. The choice of several hundreds for `k` may be satisfactory for $N$ up to 100000. Note that `k` should never exceeds $N$ in any cases.

Only the gradient of the matrix `A` will be computed when performing backward AD. The argument `k` does not require computing gradients.

### DominantSparseSymeig

```python
import DominantSparseEigenAD.symeig as symeig

symeig.setDominantSparseSymeig(A, Aadjoint_to_padjoint)
dominantsparsesymeig = symeig.DominantSparseSymeig.apply

# Usage
dominantsparsesymeig(p, k, dim)
```

This primitive is used in the case where **the matrix is assumed to be real symmetric and represented in "sparse" form as a normal function**. In current version, it will use the Lanczos algorithm to return a tuple `(eigval, eigvector)` of the **smallest** eigenvalue and corresponding eigenvector, both represented as `torch.Tensor`s. The matrix should be seen as depending on some parameters of interest.

**Parameters**: 

- **p**: `torch.Tensor` - the parameter tensor that the matrix to be diagonalized depends on.
- **k**: `int` - the number of Lanczos vectors requested. In typical applications, `k` is far less than the dimension $N$ of the matrix. The choice of several hundreds for `k` may be satisfactory for $N$ up to 100000. Note that `k` should never exceeds $N$ in any cases.
- **dim**: `int` - the dimension of the matrix. 

Only the gradient of the parameter `p` will be computed when performing backward AD. The other two arguments `k` and `dim` do not require computing gradients.

**Important Note**: To make this primitive work properly, two additional quantities, `A` and `Aadjoint_to_padjoint`, should be provided by users before the primitive is actually available in the running session: (See the second line of the code above)

- `A` is the matrix to be diagonalized. As noted above, `A` should be represented in "sparse" form as a function, which is mathematically known as a linear map that receives a vector as input and returns another vector as output. Both the input and output vectors should be `torch.Tensor`s. For example, a diagonal matrix whose diagonal elements are stored in a `torch.Tensor` `a` can be represented as a function below:

  ```python
  def diagonal(v):
      return a * v
  ```

- `Aadjoint_to_padjoint` is another function that receives the adjoint $\overline{A}$ of the matrix `A` (i.e., gradient with respect to `A` of some scalar loss) as input and returns the adjoint $\overline{p}$ of the parameter(s) `p` as output. This function receives two vectors of equal size represented as `torch.Tensor`s, `v1` and `v2`, and computes the adjoint of `p` assuming that the adjoint of `A` can be written as
  $$
  \overline{A} = v_1 v_2^T.
  $$
  For clarity, consider two simple examples:

  1. `A` can be written as the perturbation form:
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
     Hence, suppose the matrix $A_1$ is coded as a function `A1`, then the function `Aadjoint_to_padjoint` can be implemented as follows:

     ```python
     def Aadjoint_to_padjoint(v1, v2):
         return A1(v2).matmul(v1)
     ```

  2. `A` can be written as
     $$
     A = A_0 + D(\mathbf{p}).
     $$
     where $D$ is a diagonal matrix whose diagonal elements correspond to the parameters $\mathbf{p}$ (which is a vector of size equal to the dimension of the matrix `A`). Since `A` depends on $\mathbf{p}$ only through the diagonal matrix $D$, one can similarly follow the derivation above and obtains
     $$
     \overline{\mathbf{p}} = v_1 \circ v_2.
     $$
     where "$\circ$" denotes the Hadamard pairwise product. The code thus reads:

     ```python
     def Aadjoint_to_padjoint(v1, v2):
         return v1 * v2
     ```

### DominantEig

```python
from DominantSparseEigenAD.eig import DominantEig

dominanteig = DominantEig.apply

# Usage
dominanteig(A, k)
```

This primitive is used in the case where **the matrix is generally non-symmetric and represented in "normal" form as a `torch.Tensor`**. In current version, it will use the Lanczos algorithm to return a tuple `(eigval, leigvector, reigvector)` of the **largest-amplitude** eigenvalue and corresponding left and right eigenvector, all represented as `torch.Tensor`s. 

**Note**: There exist some gauge freedom regarding the normalization of the eigenvectors. For convenience, the conditions $l^T r = 1$ and $r^T r = 1$ are imposed, where $l, r$ are the left and right eigenvector, respectively.

**Note**: Since PyTorch does not have a native support of complex numbers, users of this primitive have to ensure that the desired largest-amplitude eigenvalue (hence also the corresponding eigenvectors) is real.  If this is not the case, an error will be raised.

**Parameters**:

- **A**: `torch.Tensor` - the matrix to be diagonalized.
- **k**: `int` -  the number of Lanczos vectors requested. In typical applications, `k` is far less than the dimension $N$ of the matrix `A`. The choice of several hundreds for `k` may be satisfactory for $N$ up to 100000. Note that `k` should never exceeds $N$ in any cases.

Only the gradient of the matrix `A` will be computed when performing backward AD. The argument `k` does not require computing gradients.

### DominantSparseEig

```python
import DominantSparseEigenAD.eig as eig

eig.setDominantSparseEig(A, AT, Aadjoint_to_padjoint)
dominantsparseeig = eig.DominantSparseEig.apply

# Usage
dominantsparseeig(p, k)
```

This primitive is used in the case where **the matrix is generally non-symmetric and represented in "sparse" form as a scipy [LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)**. In current version, it will use the Lanczos algorithm to return a tuple `(eigval, leigvector, reigvector)` of the **largest-amplitude** eigenvalue and corresponding left and right eigenvector, all represented as `torch.Tensor`s. The matrix should be seen as depending on some parameters of interest.

**Note**: There exist some gauge freedom regarding the normalization of the eigenvectors. For convenience, the conditions $l^T r = 1$ and $r^T r = 1$ are imposed, where $l, r$ are the left and right eigenvector, respectively.

**Note**: Since PyTorch does not have a native support of complex numbers, users of this primitive have to ensure that the desired largest-amplitude eigenvalue (hence also the corresponding eigenvectors) is real.  If this is not the case, an error will be raised.

**Parameters**: 

- **p**: `torch.Tensor` - the parameter tensor that the matrix to be diagonalized depends on.
- **k**: `int` - the number of Lanczos vectors requested. In typical applications, `k` is far less than the dimension $N$ of the matrix. The choice of several hundreds for `k` may be satisfactory for $N$ up to 100000. Note that `k` should never exceeds $N$ in any cases.

Only the gradient of the parameter `p` will be computed when performing backward AD. The argument `k` does not require computing gradients.

**Important Note**: To make this primitive work properly, three additional quantities, `A`, `AT` and `Aadjoint_to_padjoint`, should be provided by users before the primitive is actually available in the running session: (See the second line of the code above)

- `A`, `AT` are the matrix to be diagonalized and its transpose, respectively. As noted above, `A` and `AT` should be represented in "sparse" form as scipy [LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)s, which receive a vector as input and returns another vector as output.

- `Aadjoint_to_padjoint` is another function that receives the adjoint $\overline{A}$ of the matrix `A` (i.e., gradient with respect to `A` of some scalar loss) as input, and returns the adjoint $\overline{p}$ of the parameter(s) `p` as output. The input should be of the form `((u1, v1), (u2, v2), (u3, v3))`, i.e., a tuple of three pairs. The us and vs are all vectors represented as `np.ndarrays` and have size `(N,)`, where `N` is the dimension of the matrix `A`. This function computes the adjoint of `p` assuming that the adjoint of `A` can be written as
  $$
  \overline{A} = u_1 v_1^T + u_2 v_2^T + u_3 v_3^T.
  $$
  The final result of the adjoint $\overline{p}$ should be returned as a `torch.Tensor`.

  See also the primitive [DominantSparseSymeig](#dominantsparsesymeig) for some simple examples. For a more complete application, see [the VUMPS example](examples/TFIM_vumps/general.py).

## Outlook

The present interfaces of the dominant eigensolver primitives are unlikely to cover all needs in real applications. Some useful improvements and further extensions may be made in the following aspects:

- Generalization of relevant results to the case of complex numbers. (possibly with some AD tools other than PyTorch)

- Generalization to the case of multiple eigenvalues and corresponding eigenvectors.
- Implementing [reverse-mode AD of truncated SVD](https://buwantaiji.github.io/2020/01/AD-of-truncated-SVD/) by following the similar spirit.

## To cite

```bibtex
@article{PhysRevB.101.245139,
  title = {Automatic differentiation of dominant eigensolver and its applications in quantum physics},
  author = {Xie, Hao and Liu, Jin-Guo and Wang, Lei},
  journal = {Phys. Rev. B},
  volume = {101},
  issue = {24},
  pages = {245139},
  numpages = {14},
  year = {2020},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.101.245139},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.101.245139}
}
```
