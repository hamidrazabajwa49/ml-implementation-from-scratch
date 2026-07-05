"""
matrix.py
=========

A dependency-free, "from scratch" N x M matrix library built on top of
``vector.Vector``. Implements arithmetic, Gaussian elimination (REF,
determinant, inverse, rank), eigen-decomposition (QR algorithm with real
Schur block extraction for complex-conjugate eigenvalue pairs), QR
decomposition (Gram-Schmidt), and a basic SVD built from the eigenvectors
of ``A^T A``.

This single file intentionally covers what will later become three
separate challenge modules (Matrix ops / Eigenvalues / SVD) because that
is how the original code was structured. Nothing here is lost if you
later split it -- the class boundaries below (marked with section
headers) map cleanly onto "Matrix operations", "Eigenvalues &
eigenvectors", and "SVD".

Known, documented limitations (inherent to a pure-Python "from scratch"
implementation without LAPACK):
- ``eigenvalues()`` uses an *unshifted* QR algorithm. It is correct but
  can converge slowly for eigenvalues of similar magnitude, and (like
  all unshifted QR algorithms) is not recommended for large matrices.
- ``eigenvectors()`` only supports **real** eigenvalues, because
  ``Vector``/``Matrix`` store real (``int``/``float``) components only.
  Complex eigenvalues are still *correctly computed* by ``eigenvalues()``
  (e.g. for rotation matrices), but ``eigenvectors()`` will raise
  ``NotImplementedError`` if asked for the eigenvectors of a complex
  eigenvalue.
- ``svd()`` returns the first eigenvector found per eigenvalue; for
  matrices with repeated singular values the returned singular vectors
  are re-orthonormalized via Gram-Schmidt but are not guaranteed unique
  (this matches the mathematical fact that singular vectors for repeated
  singular values are only defined up to an orthogonal rotation within
  the eigenspace).

Example
-------
>>> A = Matrix([[2, 0], [0, 3]])
>>> A.determinant()
6.0
>>> A.eigenvalues()
[3.0, 2.0]
"""

from __future__ import annotations

import os
import sys
import cmath
import logging
import math
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from Vectors.vector import Vector  

Number = Union[int, float]
Scalar = Union[int, float]
EigenValue = Union[float, complex]

logger = logging.getLogger(__name__)


class MatrixError(Exception):
    """Base class for all Matrix-related errors raised by this module."""


class DimensionMismatchError(MatrixError, ValueError):
    """Raised when two matrices (or a matrix and a vector) have incompatible shapes."""


class SingularMatrixError(MatrixError, ValueError):
    """Raised when an operation (inverse, etc.) requires a non-singular matrix."""


class NotSquareError(MatrixError, ValueError):
    """Raised when an operation requires a square matrix."""


def _simplify_complex(z: complex, tol: float) -> EigenValue:
    """Collapse a complex number with negligible imaginary part to a float.

    Parameters
    ----------
    z : complex
    tol : float
        Imaginary parts with absolute value ``<= tol`` are treated as zero.

    Returns
    -------
    float or complex
    """
    if abs(z.imag) <= tol:
        return z.real
    return z


def _solve_2x2_eigs(a: float, b: float, c: float, d: float, tol: float) -> List[EigenValue]:
    """Solve for the eigenvalues of ``[[a, b], [c, d]]`` via the quadratic formula.

    Parameters
    ----------
    a, b, c, d : float
        Entries of the 2x2 matrix, row-major.
    tol : float
        Tolerance used to simplify near-real complex results.

    Returns
    -------
    list of (float or complex)
        The two eigenvalues, larger real part first.
    """
    trace = a + d
    det = a * d - b * c
    discriminant = trace * trace - 4 * det
    sqrt_disc = cmath.sqrt(discriminant)  # handles negative discriminants natively
    l1 = _simplify_complex((trace + sqrt_disc) / 2, tol)
    l2 = _simplify_complex((trace - sqrt_disc) / 2, tol)
    return [l1, l2]


class Matrix:
    """A real-valued, dense N x M matrix backed by a list of ``Vector`` rows.

    Parameters
    ----------
    data : Sequence[Sequence[Number]]
        A rectangular (all rows equal length) sequence of numeric rows.
        An empty sequence produces a 0x0 matrix.
    tol : float, optional
        Default numerical tolerance used throughout this matrix's methods
        (pivoting, singularity checks, convergence, etc). Defaults to
        ``1e-10``.

    Raises
    ------
    TypeError
        If any row is not iterable or contains a non-numeric component
        (delegated to ``Vector``).
    ValueError
        If rows have inconsistent lengths.

    Examples
    --------
    >>> Matrix([[1, 2], [3, 4]]).shape
    (2, 2)
    """

    __slots__ = ("rows", "tol", "n_rows", "n_cols")

    def __init__(self, data: Sequence[Sequence[Number]], tol: float = 1e-10) -> None:
        self.rows: List[Vector] = [Vector(row) for row in data]
        self.tol = tol

        self.n_rows: int = len(self.rows)
        self.n_cols: int = len(self.rows[0]) if self.n_rows else 0

        for i, row in enumerate(self.rows):
            if len(row) != self.n_cols:
                raise ValueError(
                    f"All rows must have the same length; row 0 has length {self.n_cols} "
                    f"but row {i} has length {len(row)}"
                )
        logger.debug("Created Matrix with shape (%d, %d)", self.n_rows, self.n_cols)


    # Basic properties / container protocol

    @property
    def shape(self) -> Tuple[int, int]:
        """Return ``(n_rows, n_cols)``."""
        return (self.n_rows, self.n_cols)

    @property
    def is_square(self) -> bool:
        """True if the matrix has an equal number of rows and columns (and is non-empty)."""
        return self.n_rows == self.n_cols and self.n_rows > 0

    def _require_square(self, operation: str) -> None:
        """Raise ``NotSquareError`` with a descriptive message if not square."""
        if self.n_rows != self.n_cols:
            raise NotSquareError(
                f"{operation} requires a square matrix, got shape {self.shape}"
            )
        if self.n_rows == 0:
            raise NotSquareError(f"{operation} is undefined for an empty (0x0) matrix")

    def _require_finite(self, operation: str) -> None:
        """Raise ``ValueError`` if any entry is NaN or infinite.

        Pivot-selection algorithms (Gaussian elimination, QR, eigen-
        solvers, power iteration) rely on ``abs(x) > tol`` comparisons to
        decide pivots. Because any comparison against NaN is ``False`` in
        Python/IEEE-754, a NaN entry silently gets treated as "already
        zero" instead of raising -- which produces a wrong, non-NaN
        answer rather than a clearly-flagged failure. We check explicitly
        instead of relying on propagation for these algorithms.
        """
        for row in self.rows:
            for x in row.components:
                if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                    raise ValueError(
                        f"{operation} requires all-finite entries; matrix contains NaN or Inf."
                    )

    def __repr__(self) -> str:
        if self.n_rows == 0:
            return "Matrix([])"
        row_strings = [str(row.components) for row in self.rows]
        rows = ",\n    ".join(row_strings)
        return f"Matrix([\n    {rows}\n])"

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Vector, List[Vector]]:
        return self.rows[idx]

    def __len__(self) -> int:
        return self.n_rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            return False
        tol = min(self.tol, other.tol)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if abs(self.rows[i].components[j] - other.rows[i].components[j]) > tol:
                    return False
        return True

    def __hash__(self):
        return None  # Matrix is mutable via row access.

    def copy(self) -> "Matrix":
        """Return a deep-enough copy (new rows, new component lists)."""
        return Matrix([row.components.copy() for row in self.rows], tol=self.tol)


    # Arithmetic

    def __add__(self, other: Union["Matrix", Scalar]) -> "Matrix":
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise DimensionMismatchError(f"Shape mismatch: {self.shape} vs {other.shape}")
            result = [x + y for x, y in zip(self, other)]
            return Matrix([v.components for v in result], tol=self.tol)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Matrix([[x + other for x in row.components] for row in self.rows], tol=self.tol)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: Union["Matrix", Scalar]) -> "Matrix":
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise DimensionMismatchError(f"Shape mismatch: {self.shape} vs {other.shape}")
            result = [x - y for x, y in zip(self, other)]
            return Matrix([v.components for v in result], tol=self.tol)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Matrix([[x - other for x in row.components] for row in self.rows], tol=self.tol)
        return NotImplemented

    def __rsub__(self, other: Union["Matrix", Scalar]) -> "Matrix":
        if isinstance(other, Matrix):
            return other.__sub__(self)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Matrix([[other - x for x in row.components] for row in self.rows], tol=self.tol)
        return NotImplemented

    def __mul__(self, other: Union["Matrix", Vector, Scalar]) -> Union["Matrix", Vector]:
        if isinstance(other, Vector):
            return self.matvec(other)

        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Matrix([[other * x for x in row.components] for row in self.rows], tol=self.tol)

        if isinstance(other, Matrix):
            if self.n_cols != other.n_rows:
                raise DimensionMismatchError(
                    f"Cannot multiply {self.shape} by {other.shape}: "
                    f"inner dimensions {self.n_cols} != {other.n_rows}"
                )
            other_cols = other.columns()  # precompute once instead of per-row
            result_rows = []
            for i in range(self.n_rows):
                row_i = self.rows[i]
                result_rows.append([row_i.dot(col_j) for col_j in other_cols])
            return Matrix(result_rows, tol=self.tol)

        return NotImplemented

    def __rmul__(self, other: Scalar) -> "Matrix":
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self.__mul__(other)
        return NotImplemented

    __matmul__ = __mul__  # allow the `@` operator as a matmul alias

    def __truediv__(self, scalar: Scalar) -> "Matrix":
        if not (isinstance(scalar, (int, float)) and not isinstance(scalar, bool)):
            raise TypeError(f"unsupported operand type(s) for /: 'Matrix' and '{type(scalar).__name__}'")
        if scalar == 0.0:
            raise ZeroDivisionError("division by zero")
        return Matrix([[x / scalar for x in row.components] for row in self.rows], tol=self.tol)

    def __neg__(self) -> "Matrix":
        return Matrix([[-x for x in row.components] for row in self.rows], tol=self.tol)

    def __pow__(self, n: int) -> "Matrix":
        """Integer matrix power via repeated squaring.

        Parameters
        ----------
        n : int
            Non-negative uses repeated multiplication; negative uses the
            inverse raised to ``abs(n)``.

        Raises
        ------
        NotSquareError
            If the matrix is not square.
        TypeError
            If ``n`` is not an integer.
        """
        self._require_square("Matrix power")
        if not isinstance(n, int):
            raise TypeError(f"matrix power exponent must be an int, got {type(n).__name__}")
        if n < 0:
            return self.inverse() ** (-n)
        result = Matrix.identity(self.n_rows)
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    def element_wise(self, func: Callable[[Number], Number]) -> "Matrix":
        """Apply ``func`` to every element and return a new Matrix."""
        return Matrix([[func(x) for x in row.components] for row in self.rows], tol=self.tol)

    def element_wise_with(self, other: "Matrix", func: Callable[[Number, Number], Number]) -> "Matrix":
        """Combine two same-shape matrices element-wise with ``func``."""
        if not isinstance(other, Matrix):
            raise TypeError(f"expected Matrix, got {type(other).__name__}")
        if self.shape != other.shape:
            raise DimensionMismatchError(f"Shape mismatch: {self.shape} vs {other.shape}")
        new_rows = [
            [func(self.rows[r].components[c], other.rows[r].components[c]) for c in range(self.n_cols)]
            for r in range(self.n_rows)
        ]
        return Matrix(new_rows, tol=self.tol)


    # Structural utilities

    def transpose(self) -> "Matrix":
        """Return the transpose (O(n*m), builds a new matrix)."""
        if self.n_rows == 0:
            return Matrix([], tol=self.tol)
        new_rows = [
            [self.rows[i].components[j] for i in range(self.n_rows)] for j in range(self.n_cols)
        ]
        return Matrix(new_rows, tol=self.tol)

    def columns(self) -> List[Vector]:
        """Return the matrix's columns as a list of Vectors."""
        n_rows, n_cols = self.shape
        return [Vector([self.rows[i].components[j] for i in range(n_rows)]) for j in range(n_cols)]

    def diagonal(self) -> "Matrix":
        """Return a square diagonal matrix built from this matrix's leading diagonal."""
        n = min(self.n_rows, self.n_cols)
        diag_mat = Matrix.zeros(n, n)
        for i in range(n):
            diag_mat.rows[i].components[i] = float(self.rows[i].components[i])
        return diag_mat

    def is_symmetric(self) -> bool:
        """True if the matrix equals its own transpose (within ``self.tol``)."""
        return self.n_rows == self.n_cols and self == self.transpose()

    def frobenius_norm(self) -> float:
        """Return the Frobenius norm ``sqrt(sum(x_ij^2))``."""
        total = sum(x * x for row in self.rows for x in row.components)
        return total ** 0.5

    @classmethod
    def zeros(cls, n_rows: int, n_cols: int) -> "Matrix":
        """Create an ``n_rows`` x ``n_cols`` matrix of zeros."""
        if n_rows < 0 or n_cols < 0:
            raise ValueError(f"zeros dimensions must be non-negative, got ({n_rows}, {n_cols})")
        return cls([[0.0] * n_cols for _ in range(n_rows)])

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        """Create an ``n`` x ``n`` identity matrix."""
        if n <= 0:
            raise ValueError(f"identity size must be positive, got {n}")
        return cls([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])


    # Gaussian elimination family: REF, determinant, inverse, rank

    def row_echelon_form(self) -> "Matrix":
        """Return the (non-reduced) row echelon form via Gaussian elimination with partial pivoting.

        Raises
        ------
        ValueError
            If the matrix contains NaN or Inf (see :meth:`_require_finite`).
        """
        self._require_finite("row_echelon_form")
        mat = self.copy()
        pivot_row = 0
        n_rows, n_cols = mat.n_rows, mat.n_cols

        for c in range(n_cols):
            pivot_index = -1
            for r in range(pivot_row, n_rows):
                if abs(mat.rows[r].components[c]) > self.tol:
                    pivot_index = r
                    break
            if pivot_index == -1:
                continue
            if pivot_index != pivot_row:
                mat.rows[pivot_row], mat.rows[pivot_index] = mat.rows[pivot_index], mat.rows[pivot_row]

            pivot_val = mat.rows[pivot_row].components[c]
            for i in range(pivot_row + 1, n_rows):
                if abs(mat.rows[i].components[c]) < self.tol:
                    continue
                factor = mat.rows[i].components[c] / pivot_val
                mat.rows[i] = mat.rows[i] - mat.rows[pivot_row] * factor

            pivot_row += 1
            if pivot_row == n_rows:
                break

        return mat

    def rank(self) -> int:
        """Return the matrix rank (number of nonzero pivot rows in REF)."""
        if self.n_rows == 0 or self.n_cols == 0:
            return 0
        ref = self.row_echelon_form()
        count = 0
        for row in ref.rows:
            if any(abs(x) > self.tol for x in row.components):
                count += 1
        return count

    def determinant(self, tol: Optional[float] = None) -> float:
        """Compute the determinant via Gaussian elimination with partial pivoting.

        Parameters
        ----------
        tol : float, optional
            Overrides ``self.tol`` for the pivot-zero check.

        Raises
        ------
        NotSquareError
            If the matrix is not square.
        """
        self._require_square("Determinant")
        self._require_finite("Determinant")
        tol = self.tol if tol is None else tol

        mat = self.copy()
        n = mat.n_rows
        swap_count = 0

        for c in range(n):
            pivot_row = -1
            for r in range(c, n):
                if abs(mat.rows[r].components[c]) > tol:
                    pivot_row = r
                    break
            if pivot_row == -1:
                return 0.0
            if pivot_row != c:
                mat.rows[c], mat.rows[pivot_row] = mat.rows[pivot_row], mat.rows[c]
                swap_count += 1

            pivot_val = mat.rows[c].components[c]
            for i in range(c + 1, n):
                factor = mat.rows[i].components[c] / pivot_val
                mat.rows[i] = mat.rows[i] - mat.rows[c] * factor

        det = (-1.0) ** swap_count
        for i in range(n):
            det *= mat.rows[i].components[i]
        return det

    def inverse(self) -> "Matrix":
        """Compute the matrix inverse via Gauss-Jordan elimination on ``[A | I]``.

        Singularity is detected during elimination itself (no separate
        determinant pre-pass), so this is a single O(n^3) sweep.

        Raises
        ------
        NotSquareError
            If the matrix is not square.
        SingularMatrixError
            If the matrix is singular (no pivot found in some column).
        """
        self._require_square("Inverse")
        self._require_finite("Inverse")
        n = self.n_rows
        identity = Matrix.identity(n)
        aug_rows = [self.rows[i].components + identity.rows[i].components for i in range(n)]
        aug = Matrix(aug_rows, tol=self.tol)

        for c in range(n):
            pivot_row = -1
            for r in range(c, n):
                if abs(aug.rows[r].components[c]) > self.tol:
                    pivot_row = r
                    break
            if pivot_row == -1:
                raise SingularMatrixError("Matrix is singular, no inverse exists.")
            if pivot_row != c:
                aug.rows[c], aug.rows[pivot_row] = aug.rows[pivot_row], aug.rows[c]

            pivot_val = aug.rows[c].components[c]
            aug.rows[c] = aug.rows[c] * (1.0 / pivot_val)

            for i in range(n):
                if i != c:
                    factor = aug.rows[i].components[c]
                    aug.rows[i] = aug.rows[i] - aug.rows[c] * factor

        return Matrix([row.components[n:] for row in aug.rows], tol=self.tol)

    def matvec(self, vec: Vector) -> Vector:
        """Compute the matrix-vector product ``A @ vec``."""
        if not isinstance(vec, Vector):
            raise TypeError(f"matvec requires a Vector, got {type(vec).__name__}")
        if self.n_cols != len(vec):
            raise DimensionMismatchError(
                f"Dimension mismatch: matrix has {self.n_cols} columns but vector has length {len(vec)}"
            )
        return Vector([row.dot(vec) for row in self.rows])


    # Eigen-decomposition

    def trace(self) -> Number:
        """Sum of the diagonal entries."""
        self._require_square("Trace")
        return sum(self.rows[i].components[i] for i in range(self.n_rows))

    def characteristic_poly(self) -> List[float]:
        """Compute the characteristic polynomial coefficients via Faddeev-LeVerrier.

        Returns coefficients ``[1, c_1, ..., c_n]`` of
        ``det(xI - A) = x^n + c_1 x^(n-1) + ... + c_n``, valid for any
        square matrix size (not just 2x2/3x3).

        Raises
        ------
        NotSquareError
            If the matrix is not square.
        """
        self._require_square("Characteristic polynomial")
        n = self.n_rows
        A = self
        M = Matrix.zeros(n, n)  # M_0
        coeffs = [1.0]  # c_0

        for k in range(1, n + 1):
            c_prev = coeffs[k - 1]
            M = A * M + c_prev * Matrix.identity(n)
            c_k = -(A * M).trace() / k
            coeffs.append(c_k)

        return coeffs

    def qr_decompose(self) -> Tuple["Matrix", "Matrix"]:
        """Classical Gram-Schmidt QR decomposition: ``A = Q R``.

        Rank-deficient columns (norm below ``1e-12`` after projection)
        yield a zero column in ``Q`` and a zero diagonal entry in ``R``;
        ``Q`` will not be fully orthogonal in that degenerate case.

        Returns
        -------
        (Matrix, Matrix)
            ``Q`` (orthonormal columns) and ``R`` (upper triangular).
        """
        self._require_finite("QR decomposition")
        n = self.n_cols
        cols = self.columns()
        q_cols: List[Vector] = []
        R = Matrix.zeros(n, n)

        for i in range(n):
            a_i = cols[i]
            v = a_i
            for j in range(i):
                q_j = q_cols[j]
                r_ji = q_j.dot(a_i)
                R.rows[j].components[i] = r_ji
                v = v - q_j * r_ji

            r_ii = v.norm()
            if r_ii < 1e-12:
                q_i = Vector([0.0] * self.n_rows)
                R.rows[i].components[i] = 0.0
            else:
                R.rows[i].components[i] = r_ii
                q_i = v * (1.0 / r_ii)
            q_cols.append(q_i)

        Q = Matrix([q.components for q in q_cols]).transpose()
        return Q, R

    def qr_algorithm(self, max_iter: int = 500, tol: Optional[float] = None) -> List[EigenValue]:
        """Compute all eigenvalues via the unshifted QR algorithm.

        Iterates ``A_{k+1} = R_k Q_k`` (from ``A_k = Q_k R_k``) until the
        matrix is quasi-upper-triangular (real Schur form): triangular
        except for 2x2 blocks on the diagonal, which correspond to pairs
        of complex-conjugate eigenvalues and are solved analytically via
        the quadratic formula.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of QR iterations.
        tol : float, optional
            Convergence / near-zero tolerance. Defaults to ``self.tol``.

        Returns
        -------
        list of (float or complex)
            One eigenvalue per matrix dimension (with algebraic
            multiplicity), in no particular guaranteed order beyond that
            imposed by the algorithm's convergence.

        Raises
        ------
        NotSquareError
            If the matrix is not square.

        Notes
        -----
        This is an *unshifted* QR algorithm: correct, but can converge
        slowly for eigenvalues of similar magnitude, and (like any
        from-scratch implementation without deflation/shifts) is not
        suitable for large or ill-conditioned matrices. For a 1x1 or 2x2
        matrix, prefer :meth:`eigenvalues`, which special-cases those
        sizes with an exact closed-form solution.
        """
        self._require_square("QR algorithm")
        self._require_finite("QR algorithm")
        tol = self.tol if tol is None else tol
        n = self.n_rows

        M = self.copy()
        converged = False
        for iteration in range(max_iter):
            Q, R = M.qr_decompose()
            M = R * Q

            # Convergence check ignores the immediate subdiagonal, since
            # legitimate 2x2 complex-eigenvalue blocks never zero it out.
            below_subdiagonal = 0.0
            for i in range(n):
                for j in range(max(0, i - 1)):
                    below_subdiagonal += M.rows[i].components[j] ** 2
            if below_subdiagonal < tol ** 2:
                converged = True
                break

        if not converged:
            logger.warning(
                "QR algorithm did not fully converge within %d iterations "
                "(matrix shape %s); returned eigenvalues may be approximate.",
                max_iter,
                self.shape,
            )

        # Extract eigenvalues from the (quasi-)upper-triangular result.
        eigenvalues: List[EigenValue] = []
        i = 0
        while i < n:
            if i == n - 1:
                eigenvalues.append(M.rows[i].components[i])
                i += 1
                continue
            subdiag = M.rows[i + 1].components[i]
            if abs(subdiag) <= tol:
                eigenvalues.append(M.rows[i].components[i])
                i += 1
            else:
                a = M.rows[i].components[i]
                b = M.rows[i].components[i + 1]
                c = M.rows[i + 1].components[i]
                d = M.rows[i + 1].components[i + 1]
                eigenvalues.extend(_solve_2x2_eigs(a, b, c, d, tol))
                i += 2

        return eigenvalues

    def eigenvalues(self, max_iter: int = 500, tol: Optional[float] = None) -> List[EigenValue]:
        """Compute all eigenvalues of this square matrix.

        Uses an exact closed-form solution for 1x1 and 2x2 matrices, and
        the QR algorithm (see :meth:`qr_algorithm`) for larger matrices.
        Correctly returns complex-conjugate pairs (e.g. for rotation
        matrices) rather than silently discarding their imaginary part.

        Raises
        ------
        NotSquareError
            If the matrix is not square.
        """
        self._require_square("Eigenvalues")
        self._require_finite("Eigenvalues")
        tol = self.tol if tol is None else tol
        n = self.n_rows

        if n == 1:
            return [float(self.rows[0].components[0])]
        if n == 2:
            a, b = self.rows[0].components
            c, d = self.rows[1].components
            return _solve_2x2_eigs(a, b, c, d, tol)
        return self.qr_algorithm(max_iter=max_iter, tol=tol)

    def eigenvectors(self, lam: EigenValue) -> List[Vector]:
        """Compute a basis for the eigenspace of real eigenvalue ``lam``.

        Solves ``(A - lam*I) x = 0`` via row echelon form and back
        substitution over the free variables.

        Parameters
        ----------
        lam : float or complex
            An eigenvalue of this matrix (typically from
            :meth:`eigenvalues`).

        Returns
        -------
        list of Vector
            One (unit-normalized) eigenvector per free variable found.
            Empty if no free variable exists (shouldn't normally happen
            for a genuine eigenvalue, but can occur due to numerical
            tolerance issues).

        Raises
        ------
        NotSquareError
            If the matrix is not square.
        NotImplementedError
            If ``lam`` has a non-negligible imaginary part: complex
            eigenvectors require complex-valued vector arithmetic, which
            is outside the scope of this real-valued Vector/Matrix
            library. Use a library such as NumPy/SciPy for that case.

        Notes
        -----
        The internal pivot/zero tolerance is scaled by this matrix's
        magnitude (``self.tol * max(1, max|A_ij|)``) rather than used as
        a raw absolute value, since floating-point error in ``A - lam*I``
        scales with the matrix's own magnitude (this matters for
        large-magnitude matrices such as ``A^T A`` inside :meth:`svd`).
        """
        self._require_square("Eigenvectors")
        self._require_finite("Eigenvectors")
        if isinstance(lam, complex) and abs(lam.imag) > self.tol:
            raise NotImplementedError(
                "Complex eigenvectors are not supported by this real-valued "
                "Vector/Matrix library. Use NumPy/SciPy for complex eigenvalue "
                f"{lam!r}."
            )
        lam = lam.real if isinstance(lam, complex) else lam

        n = self.n_rows
        scale = max((abs(x) for row in self.rows for x in row.components), default=1.0)
        working_tol = self.tol * max(1.0, scale)

        identity = Matrix.identity(n)
        M = self - (identity * lam)
        M.tol = working_tol
        R = M.row_echelon_form()

        def pivot_cols(mat: "Matrix") -> List[int]:
            pivots = []
            for i in range(mat.n_rows):
                for j in range(mat.n_cols):
                    if abs(mat.rows[i].components[j]) > working_tol:
                        pivots.append(j)
                        break
            return pivots

        pivots = pivot_cols(R)
        free = [c for c in range(n) if c not in pivots]
        if not free:
            return []

        eigenvectors_list = []
        for free_var in free:
            x = [0.0] * n
            x[free_var] = 1.0

            for i in range(R.n_rows - 1, -1, -1):
                pivot = None
                for j in range(R.n_cols):
                    if abs(R.rows[i].components[j]) > working_tol:
                        pivot = j
                        break
                if pivot is None:
                    continue
                s = sum(R.rows[i].components[j] * x[j] for j in range(pivot + 1, R.n_cols))
                x[pivot] = -s / R.rows[i].components[pivot]

            v = Vector(x)
            n_v = v.norm()
            if n_v > working_tol:
                v = Vector([comp / n_v for comp in v.components])
            eigenvectors_list.append(v)

        return eigenvectors_list

    def power_iteration(
        self,
        max_iter: int = 1000,
        tol: float = 1e-8,
        seed: Optional[int] = None,
        initial: Optional[Vector] = None,
    ) -> Tuple[float, Vector]:
        """Estimate the dominant eigenvalue/eigenvector pair via power iteration.

        Parameters
        ----------
        max_iter : int, optional
        tol : float, optional
            Convergence threshold on successive eigenvalue estimates.
        seed : int, optional
            Seed for the random initial vector, for reproducibility.
            Ignored if ``initial`` is given.
        initial : Vector, optional
            A starting vector. If omitted, a random vector is used.

        Returns
        -------
        (float, Vector)
            The dominant eigenvalue estimate and its unit eigenvector.

        Raises
        ------
        NotSquareError
            If the matrix is not square.
        ValueError
            If the matrix is the zero matrix, or the iterate's norm
            collapses to (near) zero.

        Notes
        -----
        Only converges reliably when a single real eigenvalue strictly
        dominates the others in magnitude. Does not converge for matrices
        whose dominant eigenvalues form a complex-conjugate pair of equal
        magnitude (e.g. pure rotation matrices).
        """
        self._require_square("Power iteration")
        self._require_finite("Power iteration")
        n = self.n_rows
        if all(abs(x) < self.tol for row in self.rows for x in row.components):
            raise ValueError("Power iteration is undefined for the zero matrix.")

        rng = random.Random(seed)
        if initial is not None:
            if not isinstance(initial, Vector) or len(initial) != n:
                raise ValueError(f"initial vector must be a length-{n} Vector")
            v = initial.normalize()
        else:
            vec = Vector([0.0] * n)
            attempts = 0
            while vec.norm() < self.tol:
                vec = Vector([rng.random() for _ in range(n)])
                attempts += 1
                if attempts > 100:  # pragma: no cover - astronomically unlikely
                    raise ValueError("Failed to generate a nonzero random initial vector.")
            v = vec.normalize()

        lambda_old = 0.0
        lambda_new = 0.0
        for _ in range(max_iter):
            v_new = self.matvec(v)
            lambda_new = v_new.dot(v)
            if v_new.norm() < self.tol:
                raise ValueError("Eigenvector vanished; matrix may have a zero eigenvalue.")
            v = v_new.normalize()
            if abs(lambda_new - lambda_old) < tol:
                return lambda_new, v
            lambda_old = lambda_new

        logger.warning(
            "power_iteration did not converge within %d iterations (gap=%.2e).",
            max_iter,
            abs(lambda_new - lambda_old),
        )
        return lambda_new, v

    def diagonalize(self) -> Tuple["Matrix", "Matrix"]:
        """Diagonalize as ``A = P D P^-1``.

        Raises
        ------
        NotSquareError
            If not square.
        NotImplementedError
            If any eigenvalue is complex (see :meth:`eigenvectors`).
        SingularMatrixError
            If eigenvectors do not span the space (not diagonalizable).
        """
        self._require_square("Diagonalize")
        eigvals = self.eigenvalues()
        n = len(eigvals)

        if any(isinstance(lam, complex) and abs(lam.imag) > self.tol for lam in eigvals):
            raise NotImplementedError("Complex diagonalization is not supported.")

        eigvecs = []
        for lam in eigvals:
            vlist = self.eigenvectors(lam)
            if not vlist:
                raise SingularMatrixError(
                    f"Matrix is not diagonalizable: no eigenvector found for eigenvalue {lam}."
                )
            eigvecs.append(vlist[0])

        row_matrix = Matrix([v.components for v in eigvecs])
        if abs(row_matrix.determinant()) < self.tol:
            raise SingularMatrixError("Matrix is not diagonalizable: eigenvectors are linearly dependent.")

        P = row_matrix.transpose()
        D = Matrix.zeros(n, n)
        for i, lam in enumerate(eigvals):
            D.rows[i].components[i] = float(lam.real if isinstance(lam, complex) else lam)
        return P, D

    def spectral_theorem(self) -> dict:
        """Check the spectral theorem's hypotheses/conclusions for this matrix.

        Returns
        -------
        dict
            ``{"symmetric": bool, "real_eigenvalues": bool, "orthogonal_eigenvectors": bool}``.
            Later keys are only meaningfully checked if earlier ones hold;
            the dict is returned early (with remaining keys ``False``) as
            soon as a check fails.
        """
        result = {"symmetric": False, "real_eigenvalues": False, "orthogonal_eigenvectors": False}

        if not self.is_symmetric():
            return result
        result["symmetric"] = True

        try:
            evals = self.eigenvalues()
        except (ValueError, NotImplementedError):
            return result

        real = all(not isinstance(ev, complex) or abs(ev.imag) < self.tol for ev in evals)
        result["real_eigenvalues"] = real
        if not real:
            return result

        try:
            evecs = []
            for ev in evals:
                vlist = self.eigenvectors(ev)
                if not vlist:
                    return result
                evecs.append(vlist[0])
        except (ValueError, NotImplementedError):
            return result

        orthogonal = True
        n = len(evals)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(evals[i] - evals[j]) > self.tol and abs(evecs[i].dot(evecs[j])) > self.tol:
                    orthogonal = False
                    break
            if not orthogonal:
                break

        result["orthogonal_eigenvectors"] = orthogonal
        return result


    # SVD
    
    @staticmethod
    def _gram_schmidt(vectors: List[Vector], tol: float) -> List[Vector]:
        """Classical Gram-Schmidt orthonormalization.

        Vectors that become numerically zero after projecting out the
        already-processed directions are replaced with a zero vector
        (rank-deficient / repeated-eigenvalue degeneracy).
        """
        ortho: List[Vector] = []
        for v in vectors:
            w = v
            for u in ortho:
                if u.norm() > tol:
                    w = w - u * u.dot(w)
            norm_w = w.norm()
            if norm_w > tol:
                ortho.append(w * (1.0 / norm_w))
            else:
                ortho.append(Vector([0.0] * len(v)))
        return ortho

    def svd(self) -> Tuple["Matrix", "Matrix", "Matrix"]:
        """Compute the thin (economy) Singular Value Decomposition ``A = U Sigma Vt``.

        Built from the eigendecomposition of ``A^T A`` (always real,
        symmetric, positive semi-definite, so this is numerically sound
        even though the underlying eigensolver is a basic QR algorithm).

        Uses the *thin* convention with ``k = min(m, n)``, which is the
        shape combination that always reconstructs correctly for
        non-square matrices: ``U (m x k) @ Sigma (k x k) @ Vt (k x n)``.

        Returns
        -------
        (Matrix, Matrix, Matrix)
            ``U`` (m x k), ``Sigma`` (k x k, diagonal), ``Vt`` (k x n),
            where ``k = min(m, n)``.
        """
        m, n = self.shape
        k = min(m, n)
        G = self.transpose() * self  # n x n, symmetric PSD

        evals = G.eigenvalues()
        evecs = []
        for lam in evals:
            vec_list = G.eigenvectors(lam)
            evecs.append(vec_list[0] if vec_list else Vector([0.0] * n))

        # Sort eigenvalues descending, keep eigenvectors matched, keep only
        # the first k (the rest correspond to a rank-deficient null space).
        pairs = sorted(range(len(evals)), key=lambda i: evals[i], reverse=True)[:k]
        sorted_sigmas = [max(0.0, evals[i]) ** 0.5 for i in pairs]
        sorted_evecs = [evecs[i] for i in pairs]

        # Re-orthonormalize: guards against numerically-imperfect eigenvectors,
        # especially for repeated/near-repeated singular values.
        sorted_evecs = self._gram_schmidt(sorted_evecs, self.tol)

        Vt = Matrix([v.components for v in sorted_evecs])  # k x n

        Sigma = Matrix.zeros(k, k)
        for i in range(k):
            Sigma.rows[i].components[i] = sorted_sigmas[i]

        V_cols = Vt.transpose().columns()  # k columns, each length n
        U_cols = []
        for i in range(k):
            Av = self.matvec(V_cols[i])
            norm_Av = Av.norm()
            u_i = Av * (1.0 / norm_Av) if norm_Av > 1e-12 else Vector([0.0] * m)
            U_cols.append(u_i)
        U_cols = self._gram_schmidt(U_cols, self.tol) if U_cols else U_cols

        U = Matrix([u.components for u in U_cols]).transpose() if U_cols else Matrix.zeros(m, 0)
        return U, Sigma, Vt

    @staticmethod
    def reconstruct(U: "Matrix", Sigma: "Matrix", Vt: "Matrix") -> "Matrix":
        """Reconstruct ``A`` (or a low-rank approximation) as ``U @ Sigma @ Vt``."""
        return U * Sigma * Vt

    def low_rank_approx(
        self,
        k: int,
        U: Optional["Matrix"] = None,
        Sigma: Optional["Matrix"] = None,
        Vt: Optional["Matrix"] = None,
    ) -> "Matrix":
        """Return the best rank-``k`` approximation of this matrix (Eckart-Young).

        Parameters
        ----------
        k : int
            Target rank; must be between 0 and ``min(shape)``.
        U, Sigma, Vt : Matrix, optional
            A precomputed SVD (from :meth:`svd`) to avoid recomputation.
        """
        m, n = self.shape
        if k < 0 or k > min(m, n):
            raise ValueError(f"k must be between 0 and {min(m, n)}, got {k}")
        if U is None or Sigma is None or Vt is None:
            U, Sigma, Vt = self.svd()

        full_k = Sigma.n_rows  # = min(m, n)
        Sigma_k = Matrix.zeros(full_k, full_k)
        for i in range(full_k):
            Sigma_k.rows[i].components[i] = Sigma.rows[i].components[i] if i < k else 0.0
        return U * Sigma_k * Vt

    def compression_ratio(self, k: int) -> dict:
        """Compute storage statistics for a rank-``k`` SVD approximation."""
        n_rows, n_cols = self.shape
        if k < 0 or k > min(n_rows, n_cols):
            raise ValueError(f"k must be between 0 and {min(n_rows, n_cols)}, got {k}")

        original = n_rows * n_cols
        compressed = n_rows * k + k + k * n_cols
        ratio = compressed / original if original else 0.0

        return {
            "ratio": ratio,
            "original_elements": original,
            "compressed_elements": compressed,
            "space_saved_percent": (1 - ratio) * 100 if ratio < 1 else 0,
        }

    @classmethod
    def image_compression_demo(cls) -> None:
        """Print a walkthrough of SVD-based low-rank image compression on a toy image."""
        pattern = [
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
            [0, 0, 0, 0, 50, 50, 0, 0, 0, 0],
        ]

        img = cls(pattern)
        m, n = img.shape
        max_k = min(m, n)

        print("=" * 60)
        print("SVD IMAGE COMPRESSION DEMO")
        print("=" * 60)
        print(f"\nOriginal image ({m}x{n}):")
        for row in img.rows:
            print("  " + " ".join(f"{int(x):3d}" for x in row.components))

        print("\nComputing SVD...")
        U, Sigma, Vt = img.svd()
        print("Done.\n")

        for k in [1, 2, 3, 5, max_k]:
            print(f"{'-' * 40}")
            print(f"Rank-{k} Approximation")
            print(f"{'-' * 40}")

            approx = img.low_rank_approx(k, U=U, Sigma=Sigma, Vt=Vt)
            print("Reconstruction:")
            for row in approx.rows:
                vals = [max(0, min(255, int(round(x)))) for x in row.components]
                print("  " + " ".join(f"{v:3d}" for v in vals))

            stats = img.compression_ratio(k)
            print(f"Compression ratio: {stats['ratio']:.3f}")
            print(f"Space saved: {stats['space_saved_percent']:.1f}%")
            print(f"Storage: {stats['compressed_elements']} vs {stats['original_elements']} elements")
            print(f"Reconstruction error (Frobenius): {(img - approx).frobenius_norm():.2f}")

        print(f"\n{'=' * 60}")
        print(f"Key insight: with k={max_k}, the image is perfectly reconstructed (ratio=1.0)")
        print("Lower k values trade quality for storage savings.")
        print("=" * 60)
