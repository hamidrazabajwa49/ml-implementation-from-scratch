# Matrix — Pure Python Matrix Library

## Overview

`matrix.py` is a pure Python implementation of an m×n matrix with full linear algebra support, built from scratch as **Project 2** of the *Engineering Redemption Arc* — a structured 60-project ML engineering curriculum covering mathematical foundations through production machine learning systems.

The module provides a `Matrix` class that builds directly on the `Vector` class from Project 1. It covers arithmetic operations, decompositions (LU via row reduction, QR, SVD), eigenvalue computation, matrix inversion, diagonalization, and a practical SVD-based image compression demo — all without NumPy.

---

## Project Structure

```
├── Vectors/
│   └── vector.py          # Vector primitive (Project 1 — required dependency)
└── Matrix/
    └── matrix.py          # Matrix class — all logic lives here
```

`matrix.py` dynamically resolves its parent directory at runtime to import `Vector`. Both files must be present in this folder layout for the import to succeed.

---

## Dependencies

| Requirement | Detail |
|---|---|
| Python | 3.10+ (uses `list[list]` type hint in `__init__`) |
| `math`, `os`, `sys`, `random`, `cmath` | Standard library only |
| `Vectors/vector.py` | Project 1 — local dependency |

No `pip install` required.

---

## Installation

Ensure the folder layout above is intact, then import:

```python
import sys
sys.path.insert(0, "/path/to/")

from Matrix.matrix import Matrix
from Vectors.vector import Vector
```

---

## Usage

### Construction

```python
A = Matrix([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
```

Each row must have the same length. Rows are stored internally as `Vector` objects. An optional `tol` parameter (default `1e-10`) controls the numerical zero threshold used throughout the class.

```python
A = Matrix([[1, 2], [3, 4]], tol=1e-8)
```

---

### Properties and Inspection

```python
A.shape          # (n_rows, n_cols)
A.n_rows         # int
A.n_cols         # int
A[0]             # first row as Vector
A[0][1]          # element at row 0, col 1

for row in A:    # iterate rows as Vectors
    print(row)
```

---

### Arithmetic

```python
A + B            # element-wise addition
A - B            # element-wise subtraction
A * B            # matrix multiplication
A * 3            # scalar multiplication
3 * A            # scalar multiplication (reversed)
A / 2.0          # scalar division
A == B           # approximate equality (within tol)
```

Shape mismatches raise `ValueError`. Division by zero raises `ZeroDivisionError`.

---

### Matrix–Vector Multiply

```python
v = Vector([1, 0, 0])
result = A * v         # dispatches to matvec(), returns a Vector
result = A.matvec(v)   # explicit call
```

---

### Element-wise Transformations

```python
A.element_wise(lambda x: x ** 2)
A.element_wise_with(B, lambda a, b: a + b)
```

---

### Transpose and Copy

```python
At = A.transpose()
A2 = A.copy()
```

---

### Class Methods — Constructors

```python
Matrix.zeros(3, 4)    # 3×4 zero matrix
Matrix.identity(3)    # 3×3 identity matrix
```

---

### Row Echelon Form

```python
R = A.row_echelon_form()
```

Produces upper triangular form via Gaussian elimination with partial pivoting. Does not modify `A` in place.

---

### Determinant

```python
d = A.determinant()    # float; 0.0 if singular
```

Square matrices only. Uses Gaussian elimination with swap tracking.

---

### Inverse

```python
A_inv = A.inverse()
```

Uses Gauss–Jordan elimination on the augmented matrix `[A | I]`. Raises `ValueError` if the matrix is singular.

---

### Trace

```python
t = A.trace()    # sum of diagonal elements
```

Square matrices only.

---

### Columns

```python
cols = A.columns()    # list of Vector objects, one per column
```

---

### Eigenvalues

```python
evals = A.eigenvalues()    # list of floats (real eigenvalues)
```

Delegates to the QR algorithm. Square matrices only.

---

### Eigenvectors

```python
vecs = A.eigenvectors(lam)    # list of unit Vector objects for eigenvalue lam
```

Uses row echelon form to find the null space of `(A - λI)`. Returns an empty list if no eigenvector exists.

---

### Power Iteration

```python
lam, v = A.power_iteration(max_iter=1000, tol=1e-8)
```

Returns the dominant eigenvalue and its corresponding unit eigenvector. Starts from a random nonzero vector. Prints a convergence warning if `max_iter` is reached without meeting `tol`.

---

### Characteristic Polynomial

```python
coeffs = A.characteristic_poly()    # list of float coefficients
```

Implemented only for 2×2 and 3×3 matrices. Raises `NotImplementedError` for larger sizes.

---

### QR Decomposition

```python
Q, R = A.qr_decompose()
```

Uses Modified Gram–Schmidt orthogonalization. Returns orthogonal `Q` and upper triangular `R`, both as `Matrix` objects. Handles rank-deficient columns by setting the corresponding Q column to zero.

---

### QR Algorithm (Eigenvalue Solver)

```python
evals = A.qr_algorithm(max_iter=100, tol=1e-10)
```

Iteratively applies QR decomposition until the matrix converges to (approximately) upper triangular form. Returns the diagonal entries as eigenvalues. Called internally by `eigenvalues()`.

---

### Diagonalization

```python
P, D = A.diagonalize()
```

Returns `P` (matrix of eigenvectors as columns) and `D` (diagonal matrix of eigenvalues) such that `A = P D P⁻¹`. Raises `ValueError` if the matrix is not diagonalizable. Complex eigenvalues are not supported and raise `NotImplementedError`.

---

### Spectral Theorem

```python
result = A.spectral_theorem()
```

Returns a dict verifying three properties of symmetric matrices:

```python
{
    "symmetric": True,
    "real_eigenvalues": True,
    "orthogonal_eigenvectors": True
}
```

---

### SVD

```python
U, Sigma, Vt = A.svd()
```

Computes the Singular Value Decomposition via eigendecomposition of `AᵀA`. Returns:

- `U` — left singular vectors (m×k Matrix)
- `Sigma` — diagonal singular values (m×n Matrix)
- `Vt` — right singular vectors transposed (k×n Matrix)

#### Reconstruction

```python
A_reconstructed = Matrix.reconstruct(U, Sigma, Vt)
```

#### Low-Rank Approximation

```python
A_k = A.low_rank_approx(k)                          # recomputes SVD internally
A_k = A.low_rank_approx(k, U=U, Sigma=Sigma, Vt=Vt) # reuses precomputed SVD
```

Retains only the top `k` singular values.

#### Compression Ratio

```python
stats = A.compression_ratio(k)
# {
#     "ratio": float,
#     "original_elements": int,
#     "compressed_elements": int,
#     "space_saved_percent": float
# }
```

---

### Image Compression Demo

```python
Matrix.image_compression_demo()
```

A class method that constructs a synthetic 10×10 grayscale "image" (a cross pattern), computes its SVD, and prints rank-k reconstructions for `k ∈ {1, 2, 3, 5, max_rank}` alongside compression ratios and Frobenius reconstruction errors. No external image library required.

---

## Example Session

```python
from Matrix.matrix import Matrix
from Vectors.vector import Vector

A = Matrix([[4, 3], [6, 3]])

print(A.shape)           # (2, 2)
print(A.determinant())   # -6.0
print(A.trace())         # 7

A_inv = A.inverse()
I = A * A_inv
print(I)                 # approximately identity

evals = A.eigenvalues()
print(evals)             # [6.0, 1.0] (approx)

Q, R = A.qr_decompose()
U, Sigma, Vt = A.svd()
A_approx = A.low_rank_approx(1)

Matrix.image_compression_demo()
```

---

## Error Reference

| Situation | Exception |
|---|---|
| Rows of unequal length | `ValueError` |
| Shape mismatch in `+`, `-` | `ValueError` |
| Dimension mismatch in `*` | `ValueError` |
| Non-scalar passed to `/` | `TypeError` |
| Scalar division by zero | `ZeroDivisionError` |
| Determinant/trace/eigenvalues on non-square | `ValueError` |
| Inverse of singular matrix | `ValueError` |
| `matvec` dimension mismatch | `ValueError` |
| Power iteration on zero matrix | `ValueError` |
| Eigenvector vanishes in power iteration | `ValueError` |
| `characteristic_poly` on matrix larger than 3×3 | `NotImplementedError` |
| `diagonalize` with complex eigenvalues | `NotImplementedError` |
| Matrix not diagonalizable | `ValueError` |
| Invalid `k` in `compression_ratio` | `ValueError` |

---

## Design Notes

- **Built on `Vector`:** Every row is a `Vector` instance, so all Vector arithmetic (`+`, `-`, `*`, `.dot()`, `.norm()`) is reused directly without reimplementation.
- **Tolerance parameter:** `tol` is set at construction and used consistently across all numerical comparisons (pivoting, singular checks, convergence). Override per instance when working with ill-conditioned matrices.
- **No NumPy:** All decompositions — row reduction, QR via Gram–Schmidt, SVD via eigendecomposition of AᵀA — are implemented from first principles.
- **QR algorithm limitation:** Convergence to real eigenvalues only. Complex spectra are not supported.
- **SVD via AᵀA:** This is numerically less stable than Householder-based SVD for nearly singular matrices but is pedagogically direct and sufficient for the curriculum's purposes.
- **`__eq__` uses `tol`:** Floating-point matrix equality is approximate, using the stricter of the two matrices' tolerances.

---

## Roadmap Context

This module is **Project 2 of 60** in the Engineering Redemption Arc curriculum. It depends on:

- **Project 1** — `Vector` (direct dependency; rows are Vectors)

And it underpins:

- **Projects 3–4** — Eigenvalues, SVD (both fully implemented here as precursors)
- **Project 5** — Probability and statistics (covariance matrices)
- **Projects 11+** — ML model implementations (weight matrices, gradient computation, PCA)

The SVD and low-rank approximation methods implemented here are the direct algorithmic backbone of PCA and dimensionality reduction in later phases.
