"""
base.py
=======

Abstract base class for all optimizers in this package.

Two calling conventions coexist among subclasses (see the ``step()``
docstring below for the full contract):

- **First-order** optimizers (:mod:`first_order`: ``GradientDescent``,
  ``SGD``, ``Momentum``, ``RMSProp``, ``Adam``, ``AdaGrad``) accept
  ``params`` as a list of ``float``/``Vector``/``Matrix`` elements and
  mutate it **in place**, returning ``None``.
- **Second-order** optimizers (:mod:`second_order`: ``NewtonMethod``,
  ``BFGS``) operate on a flat list of floats and **return a new list**
  rather than mutating the input, since they need the whole vector at
  once to apply a (inverse) Hessian.

Both conventions are compatible with a single flat list of floats as
"params" (each list element is then a lone float, and per-element
in-place mutation degenerates to ordinary scalar arithmetic) -- this is
what lets :func:`utils.optimize` drive either family through the same
loop.
"""

from __future__ import annotations

import os
import sys
import math
from typing import Dict, List, Union

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from Vectors.vector import Vector  # type: ignore
from Matrix.matrix import Matrix  # type: ignore

Param = Union[float, Vector, Matrix]


def _check_positive(value: float, name: str) -> None:
    """Validate that ``value`` is a positive, finite, non-NaN, non-bool real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    if math.isinf(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


class Optimizer:
    """Base class for gradient-based optimizers.

    Parameters
    ----------
    lr : float, optional
        Learning rate; must be a positive, finite real number.

    Raises
    ------
    TypeError, ValueError
        If ``lr`` fails validation (see :func:`_check_positive`).

    Attributes
    ----------
    lr : float
        Learning rate.
    iterations : int
        Number of completed ``step()`` calls.
    history : list of float
        Loss values recorded via :meth:`record`.
    """

    def __init__(self, lr: float = 0.01):
        _check_positive(lr, "lr")
        self.lr = lr
        self.iterations: int = 0
        self.history: List[float] = []

    def step(self, params: List[Param], grads: List[Param]) -> Union[None, List[float]]:
        """Apply one optimization update.

        Parameters
        ----------
        params : list
            Current parameter values.
        grads : list
            Gradients corresponding positionally to ``params``.

        Returns
        -------
        None or list of float
            First-order subclasses mutate ``params`` in place and return
            ``None``. Second-order subclasses (which need the full flat
            vector to apply a Hessian) return a **new** list rather than
            mutating the input; callers must use the return value in
            that case. See the module docstring.

        Raises
        ------
        NotImplementedError
            Always, in the base class; subclasses must override this.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset iteration count, loss history, and any accumulated optimizer state."""
        self.iterations = 0
        self.history = []

    def get_config(self) -> Dict[str, float]:
        """Return this optimizer's hyperparameters as a plain dict (for logging/serialization)."""
        return {"lr": self.lr}

    def record(self, loss: float) -> None:
        """Append ``loss`` to this optimizer's history."""
        self.history.append(loss)

    def __repr__(self) -> str:
        config = ", ".join(f"{k}={v}" for k, v in self.get_config().items())
        return f"{self.__class__.__name__}({config})"

    def _zeros_like(self, x: Param) -> Param:
        """Return a zero-valued object of the same type/shape as ``x``.

        Used by stateful optimizers (SGD with momentum, RMSProp, Adam,
        AdaGrad) to initialize per-parameter accumulators (velocity,
        squared-gradient cache, etc.) matching each parameter's type.

        Raises
        ------
        TypeError
            If ``x`` is not a ``float``/``int``, ``Vector``, or ``Matrix``.
        """
        if isinstance(x, bool):
            raise TypeError(f"Unsupported parameter type: {type(x)}")
        if isinstance(x, (int, float)):
            return 0.0
        if isinstance(x, Vector):
            return Vector([0.0] * len(x.components))
        if isinstance(x, Matrix):
            return Matrix.zeros(x.n_rows, x.n_cols)
        raise TypeError(f"Unsupported parameter type: {type(x)}")
