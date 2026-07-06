"""
line_search.py
===============

One-dimensional line search routines used to choose a step size along a
given search direction: Armijo backtracking (for use inside a descent
loop, e.g. Newton's method or BFGS) and golden-section search (for
bracketed 1D minimization).

Example
-------
>>> f = lambda x: (x[0] - 3) ** 2
>>> grad = [-6.0]
>>> direction = [1.0]
>>> alpha = backtracking(f, [0.0], grad, direction)
>>> alpha > 0
True
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[List[float]], float]


def _safe_eval(f: ObjectiveFn, x: List[float], context: str) -> float:
    """Call ``f(x)``, wrapping any exception with context about where it happened."""
    try:
        return f(x)
    except Exception as e:
        raise RuntimeError(f"Objective function raised an exception during {context} at x={x}: {e}") from e


def backtracking(
    f: ObjectiveFn,
    x: List[float],
    grad: List[float],
    direction: List[float],
    alpha: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Armijo backtracking line search: shrink ``alpha`` until the sufficient-decrease condition holds.

    Finds a step size ``alpha`` such that
    ``f(x + alpha*direction) <= f(x) + c*alpha*(grad . direction)``.

    Parameters
    ----------
    f : Callable[[list], float]
        Objective function.
    x : list of float
        Current point.
    grad : list of float
        Gradient of ``f`` at ``x``.
    direction : list of float
        Search direction; must be a descent direction (``grad . direction < 0``).
    alpha : float, optional
        Initial (largest) step size to try; must be positive.
    rho : float, optional
        Shrink factor per failed trial, in ``(0, 1)``.
    c : float, optional
        Sufficient-decrease (Armijo) constant, in ``(0, 1)``; 1e-4 is the
        standard textbook default.
    max_iter : int, optional
        Maximum number of shrink trials.

    Returns
    -------
    float
        A step size satisfying the Armijo condition, or (if ``max_iter``
        is exhausted) the smallest step size tried, with a warning logged.

    Raises
    ------
    ValueError
        If ``alpha``/``rho``/``c``/``max_iter`` are out of range, lengths
        mismatch, or ``direction`` is not a descent direction.
    RuntimeError
        If ``f`` raises an exception.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1)")
    if not (0.0 < c < 1.0):
        raise ValueError("c must be in (0, 1)")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    if len(x) != len(grad) or len(x) != len(direction):
        raise ValueError("x, grad, and direction must have the same length")

    f0 = _safe_eval(f, x, "initial evaluation")
    slope = sum(g * d for g, d in zip(grad, direction))

    if slope >= 0.0:
        raise ValueError(
            f"direction is not a descent direction (slope={slope:.6e} >= 0)"
        )

    for _ in range(max_iter):
        x_new = [xi + alpha * di for xi, di in zip(x, direction)]
        if _safe_eval(f, x_new, "trial evaluation") <= f0 + c * alpha * slope:
            return alpha
        alpha *= rho

    logger.warning(
        "backtracking line search did not satisfy the Armijo condition within "
        "%d iterations; returning the smallest step size tried (alpha=%.3e). "
        "The optimizer may make slow or no progress this step.",
        max_iter, alpha,
    )
    return alpha


def golden_section(
    f: ObjectiveFn,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> Dict:
    """Golden-section search for the minimum of a unimodal 1D function on ``[a, b]``.

    Parameters
    ----------
    f : Callable[[float], float]
        Objective function (unimodal on ``[a, b]`` -- a fundamental
        requirement of golden-section search; behavior is undefined
        otherwise).
    a, b : float
        Bracket endpoints; must be finite with ``a < b``.
    tol : float, optional
        Bracket-width convergence tolerance; must be positive.
    max_iter : int, optional
        Maximum number of shrink iterations.

    Returns
    -------
    dict
        Keys: ``x_min``, ``f_min``, ``bracket`` (final ``(a, b)``), ``converged``.

    Raises
    ------
    ValueError
        If ``a >= b``, either is non-finite, or ``tol``/``max_iter`` are invalid.
    RuntimeError
        If ``f`` raises an exception.
    """
    if math.isnan(a) or math.isnan(b):
        raise ValueError("a and b must not be NaN")
    if math.isinf(a) or math.isinf(b):
        raise ValueError("a and b must be finite")
    if a >= b:
        raise ValueError(f"a must be less than b, got a={a}, b={b}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")

    phi = (math.sqrt(5.0) - 1.0) / 2.0

    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _safe_eval(f, c, "bracket evaluation")
    fd = _safe_eval(f, d, "bracket evaluation")

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = _safe_eval(f, c, "shrink evaluation")
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = _safe_eval(f, d, "shrink evaluation")

    converged = abs(b - a) < tol
    if not converged:
        logger.warning(
            "golden_section did not converge to within tol=%.3e after %d iterations "
            "(final bracket width=%.3e).",
            tol, max_iter, abs(b - a),
        )

    x_min = (a + b) / 2.0
    return {
        "x_min": x_min,
        "f_min": _safe_eval(f, x_min, "final evaluation"),
        "bracket": (a, b),
        "converged": converged,
    }
