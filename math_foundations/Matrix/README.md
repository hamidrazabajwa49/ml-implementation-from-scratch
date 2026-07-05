# Matrix — N x M Matrix Library (From Scratch)

Part of the [`ml-implementation-from-scratch`](../../) curriculum —
**Phase 1: Math Foundations**.

A dependency-free, pure-Python N x M matrix library built directly on top
of `Vectors.vector.Vector`. It implements matrix arithmetic, Gaussian
elimination (row echelon form, determinant, inverse, rank),
eigendecomposition (QR algorithm with real-Schur handling of
complex-conjugate pairs), QR decomposition (Gram-Schmidt), and a from-scratch
Singular Value Decomposition built on the eigendecomposition of `AᵀA`.
NumPy is used only in the test suite, as a correctness oracle.

---

## Overview

`Matrix` wraps a list of `Vector` rows and implements:

- Full container protocol (`len`, indexing, iteration, equality, `copy`)
- Arithmetic operators: `+`, `-`, `*` (scalar, matrix-vector, matrix-matrix
  via `*` or `@`), `/`, unary `-`, integer matrix power `**`
- Structural utilities: `transpose`, `columns`, `diagonal`,
  `is_symmetric`, `frobenius_norm`, `zeros`, `identity`
- Gaussian elimination family: `row_echelon_form`, `rank`, `determinant`,
  `inverse`, `matvec`
- Eigendecomposition: `trace`, `characteristic_poly`, `qr_decompose`,
  `qr_algorithm`, `eigenvalues`, `eigenvectors`, `power_iteration`,
  `diagonalize`, `spectral_theorem`
- SVD: `svd`, `reconstruct`, `low_rank_approx`, `compression_ratio`,
  `image_compression_demo`

This single file intentionally covers what would later be three separate
modules (Matrix operations / Eigenvalues / SVD) — the section headers in
`matrix.py` map cleanly onto that split if it's ever needed.

---

## Project Structure

```
math_foundations/
└── Matrix/
    ├── matrix.py              # Matrix implementation (this module)
    ├── tests/
    │   └── test_matrix.py     # Pytest suite (NumPy cross-checked)
    └── README.md
```

`matrix.py` imports `Vector` from the sibling `Vectors` module
(`math_foundations/Vectors/vector.py`) via a relative `sys.path` insert —
both folders must live side-by-side under `math_foundations/`.

---

## Dependencies

| Component | Requires |
|---|---|
| `matrix.py` | Python 3.8+ standard library only (`math`, `cmath`, `random`, `logging`, `typing`) + `Vectors.vector.Vector` (sibling module, no NumPy) |
| `tests/test_matrix.py` | `pytest`, `numpy` (test-only, for regression checks) |

Install test dependencies:

```bash
pip install pytest numpy
```

Run the suite from `math_foundations/Matrix/`:

```bash
pytest tests/test_matrix.py -v
```

---

## Module Reference

### Exceptions

| Exception | Base | Raised when |
|---|---|---|
| `MatrixError` | `Exception` | Base class for all matrix errors |
| `DimensionMismatchError` | `MatrixError`, `ValueError` | Two matrices (or a matrix and vector) have incompatible shapes |
| `SingularMatrixError` | `MatrixError`, `ValueError` | An operation (inverse, diagonalize) requires a non-singular matrix |
| `NotSquareError` | `MatrixError`, `ValueError` | An operation requires a square matrix |

### Constructor

```python
Matrix(data: Sequence[Sequence[Number]], tol: float = 1e-10)
```

- Rows are validated by delegating to `Vector` (type errors on non-numeric
  or boolean entries propagate from there).
- Raises `ValueError` if rows have inconsistent lengths.
- An empty sequence produces a legal `0x0` matrix.
- `tol` is stored per-instance and used as the default numerical
  tolerance (pivoting, singularity checks, convergence) throughout that
  matrix's methods.

### Properties & container protocol

| Member | Behavior |
|---|---|
| `m.shape` | `(n_rows, n_cols)` |
| `m.is_square` | `True` if square and non-empty |
| `len(m)` | Number of rows |
| `m[i]` | Row `i` as a `Vector` (or a `list[Vector]` for slices) |
| `iter(m)` | Iterates over row `Vector`s |
| `m1 == m2` | Shape + element-wise equality within `min(m1.tol, m2.tol)` |
| `repr(m)` | Multi-line `Matrix([...])` |
| `m.copy()` | Deep-enough copy (new rows, new component lists) |
| `hash(m)` | Explicitly unhashable (mutable via row access) |

### Arithmetic

| Operation | Supports | Notes |
|---|---|---|
| `m1 + m2`, `m + scalar`, `scalar + m` | Matrix, scalar | `DimensionMismatchError` on shape mismatch |
| `m1 - m2`, `m - scalar`, `scalar - m` | Matrix, scalar | Same mismatch behavior |
| `m * scalar` | Scalar | Element-wise scale |
| `m * vector` | `Vector` | Delegates to `matvec` (matrix-vector product) |
| `m1 * m2`, `m1 @ m2` | Matrix | Matrix product; `@` is an alias for `*`. `DimensionMismatchError` if inner dimensions disagree |
| `m / scalar` | Scalar | `ZeroDivisionError` on `0.0`, `TypeError` on non-numeric |
| `-m` | — | Element-wise negation |
| `m ** n` | Non-negative or negative `int` | Repeated-squaring for `n >= 0`; `inverse() ** abs(n)` for `n < 0`. Requires square; `TypeError` if `n` isn't an `int` |

### Structural utilities

```python
m.transpose() -> Matrix
m.columns() -> List[Vector]
m.diagonal() -> Matrix              # square matrix built from the leading diagonal
m.is_symmetric() -> bool            # equals its own transpose, within tol
m.frobenius_norm() -> float
Matrix.zeros(n_rows, n_cols) -> Matrix
Matrix.identity(n) -> Matrix
m.element_wise(func) -> Matrix
m.element_wise_with(other, func) -> Matrix
```

### Gaussian elimination family

```python
m.row_echelon_form() -> Matrix
```
Non-reduced REF via Gaussian elimination with partial pivoting. Raises `ValueError` if the matrix contains NaN/Inf.

```python
m.rank() -> int
```
Number of nonzero pivot rows in REF.

```python
m.determinant(tol: float | None = None) -> float
```
Via Gaussian elimination with partial pivoting and sign tracking on row swaps. Requires square.

```python
m.inverse() -> Matrix
```
Gauss-Jordan elimination on `[A | I]` in a single O(n³) sweep. Raises `SingularMatrixError` if no pivot is found in some column.

```python
m.matvec(vec: Vector) -> Vector
```
Matrix-vector product. Raises `TypeError` if `vec` isn't a `Vector`, `DimensionMismatchError` on shape mismatch.

### Eigendecomposition

```python
m.trace() -> Number
m.characteristic_poly() -> List[float]
```
`characteristic_poly` uses Faddeev-LeVerrier, valid for any square size (not just 2x2/3x3). Returns `[1, c_1, ..., c_n]` for `det(xI - A) = x^n + c_1 x^(n-1) + ... + c_n`.

```python
m.qr_decompose() -> Tuple[Matrix, Matrix]
```
Classical Gram-Schmidt `A = Q R`. Rank-deficient columns yield a zero column in `Q` and zero diagonal entry in `R`.

```python
m.qr_algorithm(max_iter: int = 500, tol: float | None = None) -> List[EigenValue]
m.eigenvalues(max_iter: int = 500, tol: float | None = None) -> List[EigenValue]
```
`eigenvalues` special-cases 1x1/2x2 with a closed-form quadratic solution and falls back to the unshifted QR algorithm (`qr_algorithm`) otherwise. Correctly returns complex-conjugate pairs (e.g. for rotation matrices) via 2x2 real-Schur block extraction rather than discarding imaginary parts.

```python
m.eigenvectors(lam: float | complex) -> List[Vector]
```
Eigenspace basis for a given eigenvalue via REF + back substitution over free variables. Raises `NotImplementedError` for eigenvalues with non-negligible imaginary part (complex eigenvectors are out of scope for this real-valued library).

```python
m.power_iteration(max_iter=1000, tol=1e-8, seed=None, initial=None) -> Tuple[float, Vector]
```
Dominant eigenvalue/eigenvector via power iteration. Raises `ValueError` for the zero matrix or a vanishing iterate. Only converges reliably when a single real eigenvalue strictly dominates in magnitude.

```python
m.diagonalize() -> Tuple[Matrix, Matrix]
```
`A = P D P⁻¹`. Raises `NotImplementedError` for complex eigenvalues, `SingularMatrixError` if eigenvectors don't span the space.

```python
m.spectral_theorem() -> dict
```
Checks `{"symmetric", "real_eigenvalues", "orthogonal_eigenvectors"}` in sequence, short-circuiting (remaining keys `False`) as soon as a check fails.

### SVD

```python
m.svd() -> Tuple[Matrix, Matrix, Matrix]
```
Thin (economy) SVD `A = U Σ Vᵀ` with `k = min(m, n)`, built from the eigendecomposition of `AᵀA` (always real, symmetric, PSD). Singular vectors are re-orthonormalized via Gram-Schmidt to guard against repeated/near-repeated singular values.

```python
Matrix.reconstruct(U, Sigma, Vt) -> Matrix
m.low_rank_approx(k, U=None, Sigma=None, Vt=None) -> Matrix
m.compression_ratio(k) -> dict
Matrix.image_compression_demo() -> None
```
`low_rank_approx` gives the best rank-`k` approximation (Eckart-Young); `compression_ratio` reports storage stats (`ratio`, `original_elements`, `compressed_elements`, `space_saved_percent`) for a rank-`k` SVD; `image_compression_demo` prints an end-to-end walkthrough on a toy 10x10 image.

---

## Example Session

```python
from matrix import Matrix, SingularMatrixError, NotSquareError

A = Matrix([[2, 0], [0, 3]])
A.determinant()                # 6.0
A.eigenvalues()                 # [3.0, 2.0]

B = Matrix([[1, 2], [3, 4]])
B.transpose().rows[0].components   # [1, 3]
B.inverse() * B                     # ~= Matrix.identity(2)

C = Matrix([[4, 1], [2, 3]])
lam, v = C.power_iteration(seed=0)
lam                                  # dominant eigenvalue (~5.0)

Q, R = B.qr_decompose()             # B == Q * R

U, Sigma, Vt = B.svd()
Matrix.reconstruct(U, Sigma, Vt)    # ~= B

try:
    Matrix([[1, 2], [2, 4]]).inverse()
except SingularMatrixError as e:
    print(e)                        # Matrix is singular, no inverse exists.

try:
    Matrix([[1, 2, 3]]).determinant()
except NotSquareError as e:
    print(e)                        # Determinant requires a square matrix, got shape (1, 3)
```

---

## Design Notes

- **Built on `Vector`, not a raw list-of-lists.** Every row is a
  `Vectors.vector.Vector`, so row-level operations (`dot`, `norm`,
  arithmetic) reuse that module's tested, validated implementation
  instead of duplicating it.
- **NaN/Inf are explicitly rejected in pivot-driven algorithms**, unlike
  `Vector`, which lets them propagate. Gaussian elimination, QR, and the
  eigensolvers all decide pivots via `abs(x) > tol` comparisons; since any
  comparison against `NaN` is `False` in IEEE-754, a `NaN` entry would
  silently be treated as "already zero" and produce a wrong, non-NaN
  answer instead of a clearly flagged failure. `_require_finite()` guards
  every one of these entry points.
- **Unshifted QR algorithm** (`qr_algorithm`) is used for eigenvalues of
  matrices larger than 2x2. It is correct but converges slowly for
  eigenvalues of similar magnitude and is not recommended for large or
  ill-conditioned matrices — there is no deflation or shift strategy, by
  design, to keep the "from scratch" implementation tractable. 1x1 and
  2x2 matrices instead get an exact closed-form solution.
- **Real Schur handling of complex eigenvalues.** The QR algorithm
  converges to a quasi-upper-triangular matrix with 2x2 blocks on the
  diagonal wherever a pair of complex-conjugate eigenvalues exists (e.g.
  rotation matrices). Those blocks are solved analytically via the
  quadratic formula (`_solve_2x2_eigs`) rather than being misread as two
  real eigenvalues.
- **Eigenvectors are real-only.** Because `Vector`/`Matrix` store only
  `int`/`float` components, `eigenvectors()` raises `NotImplementedError`
  for any eigenvalue with non-negligible imaginary part, even though
  `eigenvalues()` correctly computes complex results. This is a
  documented scope boundary, not a bug.
- **Magnitude-scaled tolerance in `eigenvectors`.** The pivot/zero
  tolerance used when solving `(A - λI)x = 0` is scaled by the matrix's
  own magnitude (`self.tol * max(1, max|A_ij|)`) rather than used as a
  raw absolute value, since floating-point error in `A - λI` scales with
  the matrix's magnitude — this matters for large-magnitude matrices such
  as `AᵀA` inside `svd()`.
- **SVD singular vectors are not unique under repeated singular values.**
  `svd()` returns the first eigenvector found per eigenvalue of `AᵀA`,
  then re-orthonormalizes via Gram-Schmidt. This matches the
  mathematical fact that singular vectors for repeated singular values
  are only defined up to an orthogonal rotation within their eigenspace
  — not a limitation unique to this implementation.
- **`**` (matrix power) and `/` reject booleans** the same way `Vector`
  does, and `@` is a plain alias for `*` rather than a separate
  implementation, keeping matrix-multiply semantics in one place.
- **Per-instance `tol`.** Unlike `Vector` (which takes `tol` per-call),
  `Matrix` stores a default tolerance at construction time, since nearly
  every non-trivial method (pivoting, singularity, convergence) needs a
  consistent tolerance across a multi-step algorithm.

---

## Test Coverage

`tests/test_matrix.py` cross-checks numeric correctness against NumPy
(`np.linalg.det`, `np.linalg.inv`, `np.linalg.eig`, `np.linalg.svd`,
matrix multiplication, etc.) and organizes cases by the same sections as
the module: construction, container protocol, arithmetic, structural
utilities, Gaussian elimination (REF/rank/determinant/inverse/matvec),
eigendecomposition (QR decompose/algorithm, eigenvalues, eigenvectors,
power iteration, diagonalize, spectral theorem), and SVD (svd,
reconstruct, low-rank approximation, compression ratio).

Run with verbose output:

```bash
pytest tests/test_matrix.py -v
```

---

## Roadmap Context

`matrix.py` is the second module in **Phase 1: Math Foundations**,
building directly on `Vectors/vector.py`. It carries forward the same
conventions (explicit exception hierarchy, `tol`-parameterized numerical
safety, NumPy-only-in-tests policy, full docstring + type-hint coverage)
while introducing matrix-scale concerns — pivoting, convergence,
singularity — ahead of the from-scratch algorithm implementations (PCA,
linear regression, etc.) that will depend on it.
