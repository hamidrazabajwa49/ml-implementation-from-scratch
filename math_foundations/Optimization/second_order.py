"""
second_order.py
================

Second-order optimizers that use curvature information: Newton's method
(via a numerical or user-supplied Hessian) and BFGS (a quasi-Newton
method that builds an inverse-Hessian approximation from gradient
history alone).

Unlike :mod:`first_order`, these operate on a **flat list of floats**
and **return a new list** from ``step()`` rather than mutating the input
in place -- see ``base.py``'s module docstring for the full contract.
They use lightweight, dependency-free list-of-lists matrix helpers
(rather than :class:`matrix.Matrix`) deliberately: BFGS rebuilds its
full n x n inverse-Hessian approximation every iteration, and avoiding
``Matrix``'s per-operation validation overhead matters in that hot loop.

Example
-------
>>> f = lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2
>>> from utils import numerical_gradient
>>> bfgs = BFGS(lr=1.0)
>>> x = [0.0, 0.0]
>>> for _ in range(50):
...     x = bfgs.step(x, numerical_gradient(f, x))
>>> round(x[0], 2), round(x[1], 2)
(3.0, -1.0)
"""

from __future__ import annotations

import os
import sys
import logging
import math
from typing import Callable, List, Optional

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from Optimization.base import Optimizer, _check_positive  # type: ignore
from Optimization.utils import (  # type: ignore
    numerical_hessian, _vec_add, _vec_sub, _vec_scale, _vec_dot, _mat_vec,
)

logger = logging.getLogger(__name__)

Vec = List[float]
Mat = List[List[float]]



# Lightweight linear algebra helpers (see module docstring for rationale)

def _validate_square_system(A: Mat, b: Vec) -> None:
    """Validate that ``A`` is a square matrix matching ``b``'s length, with finite entries.

    Raises
    ------
    ValueError
        If ``A`` is ragged, not square, doesn't match ``len(b)``, or
        contains NaN/Inf.
    """
    n = len(b)
    if len(A) != n:
        raise ValueError(f"A has {len(A)} rows but b has length {n}")
    for i, row in enumerate(A):
        if len(row) != n:
            raise ValueError(f"A must be square ({n}x{n}); row {i} has {len(row)} columns")
        for j, v in enumerate(row):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                raise ValueError(f"A[{i}][{j}] is NaN/Inf")
    for i, v in enumerate(b):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError(f"b[{i}] is NaN/Inf")


def _solve_linear(A: Mat, b: Vec, tol: float = 1e-12) -> Vec:
    """Solve ``Ax = b`` via Gauss-Jordan elimination with partial pivoting.

    Deliberately implemented as a single-right-hand-side augmented solve
    (``[A | b]`` -> ``[I | x]``) rather than computing ``A^-1`` and
    multiplying: this does roughly half the arithmetic of a full matrix
    inversion (``n+1`` augmented columns instead of ``2n``), which matters
    since Newton's method calls this every iteration.

    Parameters
    ----------
    A : list of list of float
        Square coefficient matrix.
    b : list of float
        Right-hand side.
    tol : float, optional
        Base pivot-detection tolerance, scaled by the matrix's own
        magnitude (``tol * max(1, max|A_ij|)``) so it behaves sensibly
        for both very small and very large-magnitude systems (e.g.
        Hessians with large curvature).

    Returns
    -------
    list of float
        The solution ``x``, or ``[]`` if ``b`` is empty.

    Raises
    ------
    ValueError
        If ``A``/``b`` are malformed (see :func:`_validate_square_system`)
        or the system is singular (no pivot found in some column).
    """
    n = len(b)
    if n == 0:
        return []
    _validate_square_system(A, b)

    scale = max((abs(v) for row in A for v in row), default=1.0)
    working_tol = tol * max(1.0, scale)

    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if abs(M[row][col]) > working_tol:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Singular system: matrix is not invertible")
        M[col], M[pivot] = M[pivot], M[col]
        pivot_val = M[col][col]
        M[col] = [v / pivot_val for v in M[col]]
        for row in range(n):
            if row != col:
                factor = M[row][col]
                if factor != 0.0:
                    M[row] = [M[row][k] - factor * M[col][k] for k in range(n + 1)]

    return [M[i][n] for i in range(n)]


def _identity(n: int) -> Mat:
    """Return the ``n x n`` identity matrix as nested lists."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_mat(A: Mat, B: Mat) -> Mat:
    """Multiply two ``n x n`` matrices."""
    n = len(A)
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]


def _outer(u: Vec, v: Vec) -> Mat:
    """Outer product ``u @ v^T``."""
    return [[ui * vj for vj in v] for ui in u]


def _mat_scale(M: Mat, s: float) -> Mat:
    """Scale every entry of ``M`` by ``s``."""
    return [[Mij * s for Mij in row] for row in M]


def _mat_add(A: Mat, B: Mat) -> Mat:
    """Element-wise matrix addition."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_sub(A: Mat, B: Mat) -> Mat:
    """Element-wise matrix subtraction."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


class NewtonMethod(Optimizer):
    """Damped Newton's method: ``x -= lr * H^-1 @ grad``.

    Parameters
    ----------
    f : Callable[[list], float]
        Objective function, used to estimate the Hessian numerically
        each step (see :func:`utils.numerical_hessian`). Not needed if
        you only call :meth:`step_with_hessian`.
    lr : float, optional
        Damping factor ("learning rate"); ``lr=1.0`` is the classic
        (undamped) Newton step. Must be positive.
    hessian_h : float, optional
        Finite-difference step size for the numerical Hessian; must be positive.

    Raises
    ------
    ValueError
        If ``lr``/``hessian_h`` are non-positive.

    Notes
    -----
    Vanilla Newton's method assumes a positive-definite Hessian; if the
    Hessian has negative curvature (common far from a minimum, e.g. near
    a saddle point), the "Newton direction" can actually be an *ascent*
    direction. This implementation checks for that (via the direction's
    dot product with the gradient) and falls back to plain gradient
    descent -- with a logged warning -- rather than silently taking a
    step that increases the loss. A full "modified Newton" method
    (eigenvalue-clamping or Levenberg-Marquardt-style Hessian damping)
    would handle this more gracefully but is out of scope here.
    """

    def __init__(self, f: Callable[[Vec], float], lr: float = 1.0, hessian_h: float = 1e-4):
        super().__init__(lr)
        _check_positive(hessian_h, "hessian_h")
        self.f = f
        self.hessian_h = hessian_h

    @staticmethod
    def _newton_direction(hessian: Mat, grads: Vec) -> Vec:
        """Solve for the Newton direction, falling back to the raw gradient
        if the Hessian is singular or gives a non-descent direction."""
        try:
            direction = _solve_linear(hessian, grads)
        except ValueError:
            logger.warning(
                "Hessian is singular; falling back to a plain gradient-descent "
                "step for this iteration."
            )
            return grads[:]

        # A genuine Newton direction should satisfy direction . grad > 0
        # (so that -direction, the actual step taken, decreases f to first
        # order). If not, the Hessian has negative curvature here and the
        # "Newton direction" would be an ascent direction -- fall back.
        if _vec_dot(direction, grads) <= 0.0:
            logger.warning(
                "Newton direction is not a descent direction (negative curvature "
                "detected); falling back to a plain gradient-descent step for "
                "this iteration."
            )
            return grads[:]
        return direction

    def step(self, params: Vec, grads: Vec) -> Vec:
        """Compute a numerical Hessian at ``params`` and take one damped Newton step.

        Raises
        ------
        ValueError
            If ``params`` and ``grads`` have mismatched lengths.
        """
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        H = numerical_hessian(self.f, params, h=self.hessian_h)
        direction = self._newton_direction(H, grads)
        new_params = [p - self.lr * d for p, d in zip(params, direction)]
        self.iterations += 1
        return new_params

    def step_with_hessian(self, params: Vec, grads: Vec, hessian: Mat) -> Vec:
        """Take one damped Newton step using a caller-supplied (e.g. analytic) Hessian.

        Raises
        ------
        ValueError
            If ``params``, ``grads``, and ``hessian`` have mismatched dimensions.
        """
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if len(hessian) != len(params):
            raise ValueError("hessian dimensions must match params length")
        direction = self._newton_direction(hessian, grads)
        new_params = [p - self.lr * d for p, d in zip(params, direction)]
        self.iterations += 1
        return new_params


class BFGS(Optimizer):
    """BFGS quasi-Newton method: builds an inverse-Hessian approximation from gradient history.

    Parameters
    ----------
    lr : float, optional
        Step-size scalar applied to the BFGS search direction; must be positive.
    eps : float, optional
        Minimum curvature ``s . y`` required to apply an update (the
        standard BFGS curvature/"skip" condition); must be positive.
    max_step : float, optional
        Maximum Euclidean norm of a single step, to guard against
        divergence early in optimization when the inverse-Hessian
        approximation is still poor; must be positive.

    Raises
    ------
    ValueError
        If ``eps``/``max_step`` are non-positive.

    Notes
    -----
    Update formula (Nocedal & Wright, *Numerical Optimization*, eq. 6.17):
    ``H_{k+1} = H_k + (rho^2 * y.H_k.y + rho) * s s^T - rho*(H_k y s^T + s y^T H_k)``,
    where ``s = x_new - x_old``, ``y = g_new - g_old``, ``rho = 1/(s.y)``.

    The update is only applied when the curvature condition ``s . y >
    eps`` holds (**strictly positive**, not just "far from zero") --
    this is required to keep the inverse-Hessian approximation positive
    definite. Applying the update whenever ``|s . y| > eps`` (i.e. also
    allowing strongly *negative* curvature) can make ``H`` indefinite,
    which can turn the BFGS "descent" direction into an ascent direction.
    """

    def __init__(self, lr: float = 1.0, eps: float = 1e-10, max_step: float = 10.0):
        super().__init__(lr)
        _check_positive(eps, "eps")
        _check_positive(max_step, "max_step")
        self.eps = eps
        self.max_step = max_step
        self._H_inv: Optional[Mat] = None
        self._x_prev: Optional[Vec] = None
        self._g_prev: Optional[Vec] = None

    def step(self, params: Vec, grads: Vec) -> Vec:
        """Take one BFGS step.

        Raises
        ------
        ValueError
            If ``params`` and ``grads`` have mismatched lengths.
        """
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        n = len(params)

        if self._H_inv is None:
            self._H_inv = _identity(n)

        direction = _vec_scale(_mat_vec(self._H_inv, grads), -1.0)
        x_new = _vec_add(params, _vec_scale(direction, self.lr))

        step_vec = _vec_sub(x_new, params)
        step_norm = _vec_dot(step_vec, step_vec) ** 0.5
        if step_norm > self.max_step:
            x_new = _vec_add(params, _vec_scale(step_vec, self.max_step / step_norm))

        # Update H_inv using curvature from the PREVIOUS step: s = x_current -
        # x_prev, y = g_current - g_prev.
        if self._x_prev is not None:
            s = _vec_sub(params, self._x_prev)
            y = _vec_sub(grads, self._g_prev)
            sy = _vec_dot(s, y)

            # Curvature condition: only update if s.y is *strictly positive*
            # and not negligibly small -- required to keep H_inv positive
            # definite (see class docstring). Using abs(sy) here instead
            # would let negative-curvature regions corrupt H_inv.
            if sy > self.eps:
                Hy = _mat_vec(self._H_inv, y)
                yHy = _vec_dot(y, Hy)
                rho = 1.0 / sy

                term1 = _mat_scale(_outer(s, s), rho * (1.0 + rho * yHy))
                term2 = _mat_scale(_outer(s, Hy), rho)
                term3 = _mat_scale(_outer(Hy, s), rho)

                self._H_inv = _mat_sub(_mat_add(self._H_inv, term1), _mat_add(term2, term3))
            else:
                logger.debug(
                    "BFGS curvature condition failed (s.y=%.3e <= eps=%.3e); "
                    "skipping the inverse-Hessian update this iteration.",
                    sy, self.eps,
                )

        self._x_prev = params[:]
        self._g_prev = grads[:]
        self.iterations += 1
        return x_new

    def reset(self) -> None:
        super().reset()
        self._H_inv = None
        self._x_prev = None
        self._g_prev = None

    def get_config(self) -> dict:
        return {**super().get_config(), "eps": self.eps, "max_step": self.max_step}
