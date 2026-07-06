"""
first_order.py
===============

First-order (gradient-only) optimizers: plain gradient descent, SGD with
optional classic or Nesterov momentum, EMA-style momentum, RMSProp,
Adam, and AdaGrad.

These all accept ``params``/``grads`` as a list of ``float``/``Vector``/
``Matrix`` elements (one entry per parameter *tensor*, not necessarily
per scalar) and mutate ``params`` **in place** -- see ``base.py``'s
module docstring for the full calling-convention contract.

Example
-------
>>> opt = Adam(lr=0.1)
>>> params = [10.0]
>>> for _ in range(200):
...     grad = [2 * params[0]]  # d/dx x^2
...     opt.step(params, grad)
>>> round(params[0], 2)
0.0
"""

from __future__ import annotations

import os
import sys
import math
from typing import Callable, List, Union

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from Optimization.base import Optimizer, _check_positive  # type: ignore
from Vectors.vector import Vector  # type: ignore
from Matrix.matrix import Matrix  # type: ignore

Param = Union[float, Vector, Matrix]


def _apply_update(param: Param, update: Param) -> Param:
    """Return ``param - update`` as a new object of the same type."""
    return param - update


def _elementwise(a: Param, fn: Callable[[float], float]) -> Param:
    """Apply a scalar function ``fn`` element-wise to ``a`` (Vector/Matrix/float alike).

    Delegates to ``Vector.element_wise``/``Matrix.element_wise`` for
    tensor-valued parameters, so this correctly handles Matrix gradients
    (a naive fallback of just calling ``fn(a)`` on the whole Matrix
    object would, e.g., turn "square each entry" into a matrix
    *multiplication* via ``Matrix.__mul__`` -- entirely different math).
    """
    if isinstance(a, (Vector, Matrix)):
        return a.element_wise(fn)
    return fn(a)


def _elementwise2(a: Param, b: Param, fn: Callable[[float, float], float]) -> Param:
    """Apply a binary scalar function ``fn`` element-wise to two same-type/shape params."""
    if isinstance(a, Vector) and isinstance(b, Vector):
        return a.element_wise_with(b, fn)
    if isinstance(a, Matrix) and isinstance(b, Matrix):
        return a.element_wise_with(b, fn)
    return fn(a, b)


def _safe_sqrt(x: float) -> float:
    """``sqrt(max(x, 0))``: guards against tiny negative floating-point noise in accumulators
    that are mathematically guaranteed non-negative (sums of squares)."""
    return math.sqrt(max(x, 0.0))


class GradientDescent(Optimizer):
    """Vanilla (batch) gradient descent: ``param -= lr * grad``.

    Parameters
    ----------
    lr : float, optional
        Learning rate; must be positive.
    """

    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self, params: List[Param], grads: List[Param]) -> None:
        """Update ``params`` in place.

        Raises
        ------
        ValueError
            If ``params`` and ``grads`` have different lengths.
        """
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        for i, (p, g) in enumerate(zip(params, grads)):
            params[i] = _apply_update(p, self.lr * g)
        self.iterations += 1


class SGD(Optimizer):
    """Stochastic gradient descent with optional classic or Nesterov momentum and learning-rate decay.

    With ``momentum > 0``: ``v = momentum*v + lr*g; param -= v`` (classic
    "heavy ball" momentum -- velocity accumulates raw ``lr*g`` terms, as
    distinct from :class:`Momentum`'s exponential-moving-average style;
    see that class's docstring for the difference).

    With ``nesterov=True`` (requires ``momentum > 0``): uses Nesterov's
    accelerated-gradient look-ahead correction, ``param -= (momentum*v +
    lr*g)`` where ``v`` is updated the same way as classic momentum --
    matching PyTorch's ``SGD(nesterov=True)`` formulation.

    Parameters
    ----------
    lr : float, optional
    momentum : float, optional
        In ``[0, 1)``.
    decay : float, optional
        Time-based learning-rate decay: ``lr_eff = lr / (1 + decay*iterations)``.
        Must be non-negative.
    nesterov : bool, optional
        Use Nesterov's look-ahead correction. Requires ``momentum > 0``.

    Raises
    ------
    ValueError
        If ``momentum``/``decay`` are out of range, or ``nesterov=True``
        with ``momentum == 0``.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0, decay: float = 0.0,
                nesterov: bool = False):
        super().__init__(lr)
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if decay < 0.0:
            raise ValueError("decay must be non-negative")
        if nesterov and momentum == 0.0:
            raise ValueError("nesterov=True requires momentum > 0")
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
        self._velocity = None

    def step(self, params: List[Param], grads: List[Param]) -> None:
        """Update ``params`` in place.

        Raises
        ------
        ValueError
            If ``params`` and ``grads`` have different lengths.
        """
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")

        lr_eff = self.lr / (1.0 + self.decay * self.iterations) if self.decay > 0.0 else self.lr

        if self._velocity is None:
            self._velocity = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.momentum > 0.0:
                self._velocity[i] = self.momentum * self._velocity[i] + lr_eff * g
                if self.nesterov:
                    update = self.momentum * self._velocity[i] + lr_eff * g
                else:
                    update = self._velocity[i]
                params[i] = _apply_update(p, update)
            else:
                params[i] = _apply_update(p, lr_eff * g)

        self.iterations += 1

    def reset(self) -> None:
        super().reset()
        self._velocity = None

    def get_config(self) -> dict:
        return {**super().get_config(), "momentum": self.momentum, "decay": self.decay,
                "nesterov": self.nesterov}


class Momentum(Optimizer):
    """SGD with exponential-moving-average (EMA) style momentum.

    ``v = beta*v + (1-beta)*g; param -= lr*v``.

    Note
    ----
    This differs from :class:`SGD`'s ``momentum`` option: here the
    velocity is an EMA of the raw gradient (structurally similar to
    Adam's first-moment estimate, without bias correction), whereas
    ``SGD(momentum=beta)`` accumulates ``lr*g`` terms directly ("heavy
    ball" style). The two are mathematically different update rules for
    the same ``beta``, both standard in different textbooks/frameworks;
    this class exists to make that EMA-style formulation directly
    available and to illustrate the conceptual bridge to Adam.

    Parameters
    ----------
    lr : float, optional
    beta : float, optional
        EMA decay rate; must be in ``[0, 1)``.
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr)
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        self.beta = beta
        self._velocity = None

    def step(self, params: List[Param], grads: List[Param]) -> None:
        """Update ``params`` in place.

        Raises
        ------
        ValueError
            If ``params`` and ``grads`` have different lengths.
        """
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._velocity is None:
            self._velocity = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            self._velocity[i] = self.beta * self._velocity[i] + (1.0 - self.beta) * g
            params[i] = _apply_update(p, self.lr * self._velocity[i])

        self.iterations += 1

    def reset(self) -> None:
        super().reset()
        self._velocity = None

    def get_config(self) -> dict:
        return {**super().get_config(), "beta": self.beta}


class AdaGrad(Optimizer):
    """AdaGrad: per-parameter learning rate scaled by the inverse sqrt of the
    running sum of squared gradients (Duchi, Hazan & Singer, 2011).

    ``cache += g**2; param -= lr * g / (sqrt(cache) + epsilon)``.

    Note
    ----
    Because ``cache`` only accumulates (never decays), the effective
    learning rate monotonically shrinks over training -- this can cause
    premature stalling on long training runs; :class:`RMSProp` addresses
    this with an exponential decay on the accumulator instead.

    Parameters
    ----------
    lr : float, optional
    epsilon : float, optional
        Numerical-stability constant; must be positive.
    """

    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        super().__init__(lr)
        _check_positive(epsilon, "epsilon")
        self.epsilon = epsilon
        self._cache = None

    def step(self, params: List[Param], grads: List[Param]) -> None:
        """Update ``params`` in place.

        Raises
        ------
        ValueError
            If ``params`` and ``grads`` have different lengths.
        """
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._cache is None:
            self._cache = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            g_sq = _elementwise(g, lambda x: x * x)
            self._cache[i] = self._cache[i] + g_sq
            denom = _elementwise(self._cache[i], lambda x: _safe_sqrt(x) + self.epsilon)
            update = self.lr * _elementwise2(g, denom, lambda a, b: a / b)
            params[i] = _apply_update(p, update)

        self.iterations += 1

    def reset(self) -> None:
        super().reset()
        self._cache = None

    def get_config(self) -> dict:
        return {**super().get_config(), "epsilon": self.epsilon}


class RMSProp(Optimizer):
    """RMSProp: AdaGrad with an exponentially-decaying squared-gradient accumulator (Hinton, 2012).

    ``cache = beta*cache + (1-beta)*g**2; param -= lr * g / (sqrt(cache) + epsilon)``.

    Parameters
    ----------
    lr : float, optional
    beta : float, optional
        Decay rate for the squared-gradient accumulator; must be in ``[0, 1)``.
    epsilon : float, optional
        Numerical-stability constant; must be positive.
    """

    def __init__(self, lr: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__(lr)
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        _check_positive(epsilon, "epsilon")
        self.beta = beta
        self.epsilon = epsilon
        self._cache = None

    def step(self, params: List[Param], grads: List[Param]) -> None:
        """Update ``params`` in place.

        Raises
        ------
        ValueError
            If ``params`` and ``grads`` have different lengths.
        """
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._cache is None:
            self._cache = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            g_sq = _elementwise(g, lambda x: x * x)
            self._cache[i] = self.beta * self._cache[i] + (1.0 - self.beta) * g_sq
            denom = _elementwise(self._cache[i], lambda x: _safe_sqrt(x) + self.epsilon)
            update = self.lr * _elementwise2(g, denom, lambda a, b: a / b)
            params[i] = _apply_update(p, update)

        self.iterations += 1

    def reset(self) -> None:
        super().reset()
        self._cache = None

    def get_config(self) -> dict:
        return {**super().get_config(), "beta": self.beta, "epsilon": self.epsilon}


class Adam(Optimizer):
    """Adam: adaptive moment estimation (Kingma & Ba, 2015).

    ``m = beta1*m + (1-beta1)*g``; ``v = beta2*v + (1-beta2)*g**2``;
    bias-corrected ``m_hat = m/(1-beta1**t)``, ``v_hat = v/(1-beta2**t)``;
    ``param -= lr * m_hat / (sqrt(v_hat) + epsilon)``.

    Note
    ----
    This computes the bias-corrected moments explicitly and applies
    ``epsilon`` to ``sqrt(v_hat)``, matching the reference formula in
    Algorithm 1 of the paper and the default behavior of PyTorch's/
    TensorFlow's Adam. (The paper's Section 2 also describes an
    alternative, mathematically-almost-equivalent reformulation that
    folds both bias corrections into a single combined learning-rate
    scalar applied to *uncorrected* ``m``/``v`` -- but that variant
    changes the effective scale of ``epsilon``, so it does not exactly
    reproduce results from PyTorch/TensorFlow/other Adam
    implementations. This class deliberately uses the textbook/
    PyTorch-compatible form instead, since that's what "matches Adam"
    means in practice for nearly everyone comparing results.)

    Parameters
    ----------
    lr : float, optional
    beta1 : float, optional
        First-moment decay rate; must be in ``[0, 1)``.
    beta2 : float, optional
        Second-moment decay rate; must be in ``[0, 1)``.
    epsilon : float, optional
        Numerical-stability constant; must be positive.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                epsilon: float = 1e-8):
        super().__init__(lr)
        if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        _check_positive(epsilon, "epsilon")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m = None
        self._v = None

    def step(self, params: List[Param], grads: List[Param]) -> None:
        """Update ``params`` in place.

        Raises
        ------
        ValueError
            If ``params`` and ``grads`` have different lengths.
        """
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._m is None:
            self._m = [self._zeros_like(p) for p in params]
            self._v = [self._zeros_like(p) for p in params]

        self.iterations += 1
        t = self.iterations
        bias_correction1 = 1.0 - self.beta1 ** t
        bias_correction2 = 1.0 - self.beta2 ** t

        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            g_sq = _elementwise(g, lambda x: x * x)
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g_sq

            m_hat = _elementwise(self._m[i], lambda x: x / bias_correction1)
            v_hat = _elementwise(self._v[i], lambda x: x / bias_correction2)
            denom = _elementwise(v_hat, lambda x: _safe_sqrt(x) + self.epsilon)
            update = self.lr * _elementwise2(m_hat, denom, lambda a, b: a / b)
            params[i] = _apply_update(p, update)

    def reset(self) -> None:
        super().reset()
        self._m = None
        self._v = None

    def get_config(self) -> dict:
        return {**super().get_config(), "beta1": self.beta1, "beta2": self.beta2,
                "epsilon": self.epsilon}
