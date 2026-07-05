"""
special_functions.py
=====================

Dependency-free implementations of the regularized incomplete beta and
gamma functions, the numerical backbone for the CDFs of the Student-t,
chi-squared, F, gamma, and beta distributions elsewhere in this package.

Both functions use Lentz's modified continued-fraction algorithm (Numerical
Recipes, 3rd ed., sections 6.2 and 6.4), which converges quickly and
accurately across the parameter ranges typically seen in statistics (say
a, b up to a few thousand). All internal arithmetic is done in log-space
where possible to avoid overflow for large shape parameters.

Example
-------
>>> round(betainc(2.0, 3.0, 0.5), 6)
0.6875
>>> round(gammainc(2.0, 3.0), 6)
0.800852
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

_TINY = 1e-300  # Lentz's algorithm floor to avoid exact-zero division.


def _validate_shape_param(name: str, value: float) -> None:
    """Raise ``ValueError``/``TypeError`` if ``value`` is not a positive, finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    if math.isinf(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def betainc(a: float, b: float, x: float, max_iter: int = 200, tol: float = 1e-15) -> float:
    """Regularized incomplete beta function :math:`I_x(a, b)`.

    .. math::
        I_x(a, b) = \\frac{1}{B(a, b)} \\int_0^x t^{a-1} (1-t)^{b-1}\\, dt

    Parameters
    ----------
    a, b : float
        Shape parameters; must be positive and finite.
    x : float
        Upper limit of integration; must be in ``[0, 1]``.
    max_iter : int, optional
        Maximum continued-fraction iterations.
    tol : float, optional
        Convergence tolerance on successive continued-fraction convergents.

    Returns
    -------
    float
        A value in ``[0, 1]``.

    Raises
    ------
    TypeError
        If ``a``, ``b``, or ``x`` is not a real number.
    ValueError
        If ``a`` or ``b`` is non-positive/non-finite/NaN, or ``x`` is
        outside ``[0, 1]`` or NaN.
    """
    _validate_shape_param("a", a)
    _validate_shape_param("b", b)
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"x must be a real number, got {type(x).__name__}")
    if math.isnan(x):
        raise ValueError("x must not be NaN")
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Use the symmetry relation I_x(a,b) = 1 - I_{1-x}(b,a) for faster
    # convergence: the continued fraction converges quickly only for
    # x < (a+1)/(a+b+2).
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - betainc(b, a, 1.0 - x, max_iter=max_iter, tol=tol)

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a

    C = 1.0
    D = 1.0 - (a + b) * x / (a + 1.0)
    if abs(D) < _TINY:
        D = _TINY
    D = 1.0 / D
    f = D

    converged = False
    for i in range(1, max_iter + 1):
        m = i
        delta = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        D = 1.0 + delta * D
        if abs(D) < _TINY:
            D = _TINY
        C = 1.0 + delta / C
        if abs(C) < _TINY:
            C = _TINY
        D = 1.0 / D
        f *= C * D

        delta = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        D = 1.0 + delta * D
        if abs(D) < _TINY:
            D = _TINY
        C = 1.0 + delta / C
        if abs(C) < _TINY:
            C = _TINY
        D = 1.0 / D
        delta = C * D
        f *= delta

        if abs(delta - 1.0) < tol:
            converged = True
            break

    if not converged:
        logger.warning(
            "betainc(a=%s, b=%s, x=%s) did not converge within %d iterations; "
            "result may be inaccurate.",
            a, b, x, max_iter,
        )

    result = front * f
    return min(max(result, 0.0), 1.0)  # clamp tiny floating-point overshoot


def betaincc(a: float, b: float, x: float, max_iter: int = 200, tol: float = 1e-15) -> float:
    """Complement of :func:`betainc`: ``1 - I_x(a, b)``, computed directly for accuracy."""
    _validate_shape_param("a", a)
    _validate_shape_param("b", b)
    if x < 0.0 or x > 1.0 or (isinstance(x, float) and math.isnan(x)):
        raise ValueError(f"x must be in [0, 1], got {x}")
    return betainc(b, a, 1.0 - x, max_iter=max_iter, tol=tol)


def gammainc(a: float, x: float, max_iter: int = 300, tol: float = 1e-12) -> float:
    """Regularized lower incomplete gamma function :math:`P(a, x)`.

    .. math::
        P(a, x) = \\frac{1}{\\Gamma(a)} \\int_0^x t^{a-1} e^{-t}\\, dt

    Parameters
    ----------
    a : float
        Shape parameter; must be positive and finite.
    x : float
        Upper limit of integration; must be non-negative.
    max_iter : int, optional
        Maximum series/continued-fraction iterations.
    tol : float, optional
        Relative convergence tolerance.

    Returns
    -------
    float
        A value in ``[0, 1]``.

    Raises
    ------
    TypeError
        If ``a`` or ``x`` is not a real number.
    ValueError
        If ``a`` is non-positive/non-finite/NaN, or ``x`` is negative or NaN.
    """
    _validate_shape_param("a", a)
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"x must be a real number, got {type(x).__name__}")
    if math.isnan(x):
        raise ValueError("x must not be NaN")
    if x < 0.0:
        raise ValueError(f"x must be non-negative, got {x}")
    if x == 0.0:
        return 0.0
    if math.isinf(x):
        return 1.0

    if x < a + 1.0:
        return _gammainc_series(a, x, max_iter, tol)
    return 1.0 - _gammainc_cf(a, x, max_iter, tol)


def gammaincc(a: float, x: float, max_iter: int = 300, tol: float = 1e-12) -> float:
    """Regularized upper incomplete gamma function :math:`Q(a, x) = 1 - P(a, x)`."""
    return 1.0 - gammainc(a, x, max_iter=max_iter, tol=tol)


def _gammainc_series(a: float, x: float, max_iter: int, tol: float) -> float:
    """Power-series expansion for ``P(a, x)``, valid/fast-converging for ``x < a + 1``."""
    term = 1.0 / a
    total = term
    converged = False
    for n in range(1, max_iter + 1):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * tol:
            converged = True
            break
    if not converged:
        logger.warning(
            "gammainc series (a=%s, x=%s) did not converge within %d iterations.",
            a, x, max_iter,
        )
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammainc_cf(a: float, x: float, max_iter: int, tol: float) -> float:
    """Continued-fraction expansion for ``Q(a, x)``, valid/fast-converging for ``x >= a + 1``."""
    b = x + 1.0 - a
    C = 1.0 / _TINY
    D = 1.0 / b if abs(b) > _TINY else 1.0 / _TINY
    f = D
    converged = False
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        D = an * D + b
        if abs(D) < _TINY:
            D = _TINY
        C = b + an / C
        if abs(C) < _TINY:
            C = _TINY
        D = 1.0 / D
        delta = D * C
        f *= delta
        if abs(delta - 1.0) < tol:
            converged = True
            break
    if not converged:
        logger.warning(
            "gammainc continued fraction (a=%s, x=%s) did not converge within %d iterations.",
            a, x, max_iter,
        )
    return f * math.exp(-x + a * math.log(x) - math.lgamma(a))
