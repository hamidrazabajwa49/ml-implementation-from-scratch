"""
utils.py
========

Shared numerical utilities for the optimization module: finite-difference
gradient/Hessian estimation, lightweight vector helpers (used by the
flat-list-based second-order optimizers), a convergence tracker with
early-stopping support, and a generic ``optimize()`` driver loop that ties
an objective function, its gradient, and any :class:`base.Optimizer`
together.

Two calling conventions coexist in this package (see ``base.Optimizer``
for the full contract):

- **First-order optimizers** (``GradientDescent``, ``SGD``, ``Momentum``,
  ``RMSProp``, ``Adam``, ``AdaGrad``) accept ``params`` as a list of
  ``float``/``Vector``/``Matrix`` elements and mutate it **in place**
  (``step()`` returns ``None``).
- **Second-order optimizers** (``NewtonMethod``, ``BFGS``) operate on a
  single **flat list of floats** and **return a new list** rather than
  mutating the input (they need the whole vector at once to form/apply a
  Hessian or its inverse).

``optimize()`` below handles both transparently: ``result =
optimizer.step(x, grads); if result is not None: x = result``.

Example
-------
>>> f = lambda x: x[0] ** 2 + x[1] ** 2
>>> grad_f = lambda x: numerical_gradient(f, x)
>>> from first_order import GradientDescent
>>> result = optimize(f, grad_f, [3.0, 4.0], GradientDescent(lr=0.1), max_iter=200)
>>> result['loss'] < 1e-4
True
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

Vec = List[float]
Mat = List[List[float]]
ObjectiveFn = Callable[[Vec], float]
GradientFn = Callable[[Vec], Vec]



# Finite-difference derivatives

def numerical_gradient(f: ObjectiveFn, x: Vec, h: float = 1e-5) -> Vec:
    """Central-difference gradient estimate: ``(f(x+h*e_i) - f(x-h*e_i)) / (2h)`` per coordinate.

    Parameters
    ----------
    f : Callable[[list], float]
        Scalar objective function.
    x : list of float
        Point to evaluate the gradient at; must be non-empty.
    h : float, optional
        Finite-difference step size; must be positive.

    Returns
    -------
    list of float
        Same length as ``x``.

    Raises
    ------
    ValueError
        If ``x`` is empty or ``h`` is non-positive.

    Notes
    -----
    Cost: ``2 * len(x)`` evaluations of ``f``. Central differences are
    O(h^2)-accurate, versus O(h) for one-sided (forward/backward)
    differences -- worth the extra evaluation per coordinate.

    Safe to call with a NumPy array for ``x`` (e.g. when interoperating
    with ``scipy.optimize``): the input is defensively copied to a plain
    list first, since ``x[:]`` is a *view* (not a copy) for NumPy arrays,
    which would otherwise cause in-place perturbation of one coordinate
    to silently corrupt the caller's original array.
    """
    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    x = list(x)  # defensive copy: x[:] is a VIEW (not a copy) for numpy
    # arrays, so in-place mutation below would silently corrupt the
    # caller's array if we didn't normalize to a plain list first.

    grad = []
    for i in range(len(x)):
        x_fwd = x[:]
        x_bwd = x[:]
        x_fwd[i] += h
        x_bwd[i] -= h
        grad.append((f(x_fwd) - f(x_bwd)) / (2.0 * h))
    return grad


def numerical_hessian(f: ObjectiveFn, x: Vec, h: float = 1e-4) -> Mat:
    """Central-difference Hessian estimate.

    Uses the standard 3-point second-derivative formula for diagonal
    entries (``(f(x+h) - 2f(x) + f(x-h)) / h^2``) and the standard 4-point
    mixed-partial formula for off-diagonal entries, and exploits the
    Hessian's symmetry (``H[i][j] == H[j][i]``) to only evaluate each
    off-diagonal pair once.

    Parameters
    ----------
    f : Callable[[list], float]
    x : list of float
        Must be non-empty.
    h : float, optional
        Finite-difference step size; must be positive.

    Returns
    -------
    list of list of float
        An ``n x n`` symmetric matrix (as nested lists).

    Raises
    ------
    ValueError
        If ``x`` is empty or ``h`` is non-positive.

    Notes
    -----
    Cost: ``1 + 2n + 4*n(n-1)/2 = 2n^2 + 1`` evaluations of ``f`` -- about
    half of the ``4n^2`` a naive "apply the mixed-partial formula to every
    (i, j) pair including the diagonal" implementation would use (that
    naive approach also silently uses an inconsistent effective step size
    of ``2h`` rather than ``h`` on the diagonal, and evaluates ``f(x)``
    twice per diagonal entry for no reason). This is still O(n^2)
    evaluations overall, which is the fundamental cost of a from-scratch
    numerical Hessian without analytic derivatives or automatic
    differentiation -- expect this to be slow for more than a few dozen
    parameters.
    """
    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    n = len(x)
    if n == 0:
        raise ValueError("x must be non-empty")
    x = list(x)  # defensive copy: see numerical_gradient for why this matters
    # (x[:] is a view, not a copy, for numpy arrays).
    if n > 50:
        logger.warning(
            "numerical_hessian called with n=%d parameters: this requires "
            "~%d function evaluations (O(n^2)) and may be slow. Consider "
            "an analytic Hessian or a quasi-Newton method (BFGS) instead.",
            n, 2 * n * n + 1,
        )

    H = [[0.0] * n for _ in range(n)]
    f0 = f(x)

    for i in range(n):
        x_p = x[:]
        x_p[i] += h
        x_m = x[:]
        x_m[i] -= h
        H[i][i] = (f(x_p) - 2.0 * f0 + f(x_m)) / (h * h)

        for j in range(i + 1, n):
            x_pp = x[:]; x_pp[i] += h; x_pp[j] += h
            x_pm = x[:]; x_pm[i] += h; x_pm[j] -= h
            x_mp = x[:]; x_mp[i] -= h; x_mp[j] += h
            x_mm = x[:]; x_mm[i] -= h; x_mm[j] -= h
            val = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * h * h)
            H[i][j] = val
            H[j][i] = val

    return H


'''
Lightweight flat-vector helpers (used by second_order.py; deliberately
plain-list based rather than Vector/Matrix, to avoid per-call validation
overhead in BFGS's hot inner loop - see second_order.py's module docstring)
'''


def _check_same_length(a: Vec, b: Vec, op: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{op}: length mismatch ({len(a)} vs {len(b)})")


def _vec_add(a: Vec, b: Vec) -> Vec:
    _check_same_length(a, b, "_vec_add")
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_sub(a: Vec, b: Vec) -> Vec:
    _check_same_length(a, b, "_vec_sub")
    return [ai - bi for ai, bi in zip(a, b)]


def _vec_scale(a: Vec, s: float) -> Vec:
    return [ai * s for ai in a]


def _vec_dot(a: Vec, b: Vec) -> float:
    _check_same_length(a, b, "_vec_dot")
    return sum(ai * bi for ai, bi in zip(a, b))


def _vec_norm(a: Vec) -> float:
    return math.sqrt(sum(ai * ai for ai in a))


def _mat_vec(M: Mat, v: Vec) -> Vec:
    if M and len(M[0]) != len(v):
        raise ValueError(f"_mat_vec: matrix has {len(M[0])} columns but vector has length {len(v)}")
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]



# Convergence tracking

class ConvergenceTracker:
    """Tracks loss history and signals early stopping when improvement plateaus.

    Parameters
    ----------
    tol : float, optional
        Minimum decrease in loss (vs. the best value seen so far) to
        count as "improvement"; must be positive.
    patience : int, optional
        Number of consecutive non-improving updates to tolerate before
        :meth:`update` signals a stop; must be at least 1.

    Raises
    ------
    ValueError
        If ``tol`` is non-positive or ``patience`` is less than 1.
    """

    def __init__(self, tol: float = 1e-6, patience: int = 10):
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}")
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        self.tol = tol
        self.patience = patience
        self.history: List[float] = []
        self._no_improve = 0
        self._best = float("inf")

    def update(self, loss: float) -> bool:
        """Record a new loss value.

        Returns
        -------
        bool
            True if there has been no improvement (of at least ``tol``)
            for ``patience`` consecutive calls -- a signal the caller
            should consider stopping early.
        """
        self.history.append(loss)
        if loss < self._best - self.tol:
            self._best = loss
            self._no_improve = 0
        else:
            self._no_improve += 1
        return self._no_improve >= self.patience

    def converged(self) -> bool:
        """True if the last two recorded losses differ by less than ``tol``.

        Note
        ----
        This is a simple two-point check; a loss that oscillates within a
        band smaller than ``tol`` but isn't truly converging could trigger
        a false positive. For a stronger signal, prefer :meth:`update`'s
        patience-based early stopping.
        """
        if len(self.history) < 2:
            return False
        return abs(self.history[-1] - self.history[-2]) < self.tol

    def reset(self) -> None:
        """Clear all recorded history and internal state."""
        self.history = []
        self._no_improve = 0
        self._best = float("inf")

    def plot_ascii(self, width: int = 60, height: int = 12) -> None:
        """Print a crude ASCII bar chart of the loss history to stdout.

        Parameters
        ----------
        width : int, optional
            Unused directly (kept for API stability / future use); the
            plotted width is currently the number of recorded losses.
        height : int, optional
            Number of vertical rows; must be at least 2.

        Raises
        ------
        ValueError
            If ``height`` is less than 2 (needed to avoid a division by
            zero when computing row thresholds).
        """
        if height < 2:
            raise ValueError(f"height must be at least 2, got {height}")
        if not self.history:
            print("No history to plot.")
            return
        losses = self.history
        lo, hi = min(losses), max(losses)
        span = hi - lo if hi != lo else 1.0
        print(f"\n  Loss curve  (iter 0 \u2192 {len(losses) - 1})")
        print(f"  max: {hi:.6f}")
        for row in range(height):
            threshold = hi - (row / (height - 1)) * span
            line = ""
            for val in losses:
                line += "\u2588" if val >= threshold - span / height else " "
            print(f"  |{line}")
        print(f"  min: {lo:.6f}")
        print(f"  {'\u2500' * len(losses)}")



# Driver loop

def optimize(
    f: ObjectiveFn,
    grad_f: GradientFn,
    x0: Vec,
    optimizer,
    max_iter: int = 1000,
    tol: float = 1e-6,
    patience: int = 20,
    verbose: bool = False,
) -> dict:
    """Run ``optimizer`` on ``f`` starting from ``x0`` until convergence or ``max_iter``.

    Stops early on either of two conditions: the gradient norm drops
    below ``tol``, or the loss fails to improve (by at least ``tol``) for
    ``patience`` consecutive iterations (plateau detection via
    :class:`ConvergenceTracker`).

    Parameters
    ----------
    f : Callable[[list], float]
        Objective function.
    grad_f : Callable[[list], list]
        Gradient of ``f`` (analytic or from :func:`numerical_gradient`).
    x0 : list of float
        Starting point; must be non-empty.
    optimizer
        Any object with a ``step(params, grads)`` method following either
        the in-place-mutation or return-new-params convention (see the
        module docstring), plus a ``record(loss)`` method and ``history``
        attribute (any :class:`base.Optimizer` subclass qualifies).
    max_iter : int, optional
        Maximum number of iterations; must be at least 1.
    tol : float, optional
        Gradient-norm and plateau-detection tolerance; must be positive.
    patience : int, optional
        Iterations to tolerate without improvement before stopping early;
        must be at least 1.
    verbose : bool, optional
        If True, print progress roughly every ``max_iter // 10`` iterations.

    Returns
    -------
    dict
        Keys: ``x`` (final point), ``loss``, ``iterations``, ``history``
        (loss per iteration), ``converged`` (bool: final gradient norm < tol).

    Raises
    ------
    ValueError
        If ``x0`` is empty, or ``max_iter``/``tol``/``patience`` are invalid.
    TypeError
        If ``optimizer`` lacks a callable ``step`` method.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if len(x0) == 0:
        raise ValueError("x0 must be non-empty")
    if not callable(getattr(optimizer, "step", None)):
        raise TypeError("optimizer must have a callable 'step(params, grads)' method")

    x = x0[:]
    tracker = ConvergenceTracker(tol=tol, patience=patience)
    iterations = 0
    stop_reason = "max_iter reached"

    for i in range(max_iter):
        iterations = i + 1
        loss = f(x)
        grads = grad_f(x)
        plateaued = tracker.update(loss)
        optimizer.record(loss)

        if verbose and i % max(1, max_iter // 10) == 0:
            gnorm = _vec_norm(grads)
            print(f"  iter {i:5d}  loss={loss:.8f}  |grad|={gnorm:.2e}")

        if _vec_norm(grads) < tol:
            stop_reason = f"gradient norm < {tol}"
            if verbose:
                print(f"  converged at iter {i}  ({stop_reason})")
            break

        if plateaued:
            stop_reason = f"no improvement for {patience} iterations (plateau)"
            if verbose:
                print(f"  stopped early at iter {i}  ({stop_reason})")
            break

        # step() may mutate x in-place (first-order optimizers) or return a
        # new x (second-order optimizers) -- see module docstring.
        result = optimizer.step(x, grads)
        if result is not None:
            x = result

    final_grad_norm = _vec_norm(grad_f(x))
    logger.debug("optimize() finished after %d iterations: %s", iterations, stop_reason)
    return {
        "x": x,
        "loss": f(x),
        "iterations": iterations,
        "history": optimizer.history[:],
        "converged": final_grad_norm < tol,
        "stop_reason": stop_reason,
    }
