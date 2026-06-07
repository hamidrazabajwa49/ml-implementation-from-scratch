import os
import sys
import math
from typing import List, Union


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Optimization.base import Optimizer
from Vectors.vector import Vector
from Matrix.matrix import Matrix


def _apply_update(param: Union[Vector, Matrix, float],
                update: Union[Vector, Matrix, float]) -> Union[Vector, Matrix, float]:
    """Return param - update as a new object of the same type."""
    return param - update


def _zeros_like_components(p: Union[Vector, Matrix, float]) -> list:
    if isinstance(p, Vector):
        return [0.0] * len(p)
    if isinstance(p, Matrix):
        return [[0.0] * p.n_cols for _ in range(p.n_rows)]
    return 0.0


def _vec_op(a: Union[Vector, float], fn) -> Union[Vector, float]:
    if isinstance(a, Vector):
        return Vector([fn(a[i]) for i in range(len(a))])
    return fn(a)


def _vec_op2(a: Union[Vector, float], b: Union[Vector, float], fn) -> Union[Vector, float]:
    if isinstance(a, Vector) and isinstance(b, Vector):
        return Vector([fn(a[i], b[i]) for i in range(len(a))])
    return fn(a, b)


def _safe_sqrt(x: float) -> float:
    return math.sqrt(max(x, 0.0))


class GradientDescent(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self, params: List[Union[float, Vector, Matrix]],
            grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        for i, (p, g) in enumerate(zip(params, grads)):
            params[i] = _apply_update(p, self.lr * g)
        self.iterations += 1


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.0, decay: float = 0.0):
        super().__init__(lr)
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if decay < 0.0:
            raise ValueError("decay must be non-negative")
        self.momentum = momentum
        self.decay = decay
        self._velocity = None

    def step(self, params: List[Union[float, Vector, Matrix]],
            grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")

        lr_eff = self.lr / (1.0 + self.decay * self.iterations) if self.decay > 0.0 else self.lr

        if self._velocity is None:
            self._velocity = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.momentum > 0.0:
                # standard: v = momentum*v + lr*g  ;  p = p - v
                self._velocity[i] = self.momentum * self._velocity[i] + lr_eff * g
                params[i] = _apply_update(p, self._velocity[i])
            else:
                params[i] = _apply_update(p, lr_eff * g)

        self.iterations += 1

    def reset(self):
        super().reset()
        self._velocity = None

    def get_config(self):
        return {**super().get_config(), "momentum": self.momentum, "decay": self.decay}


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr)
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        self.beta = beta
        self._velocity = None

    def step(self, params: List[Union[float, Vector, Matrix]],
            grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._velocity is None:
            self._velocity = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            self._velocity[i] = self.beta * self._velocity[i] + (1.0 - self.beta) * g
            params[i] = _apply_update(p, self.lr * self._velocity[i])

        self.iterations += 1

    def reset(self):
        super().reset()
        self._velocity = None

    def get_config(self):
        return {**super().get_config(), "beta": self.beta}


class RMSProp(Optimizer):
    def __init__(self, lr: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__(lr)
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.beta = beta
        self.epsilon = epsilon
        self._cache = None

    def step(self, params: List[Union[float, Vector, Matrix]],
            grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._cache is None:
            self._cache = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            g_sq = _vec_op(g, lambda x: x * x)
            self._cache[i] = self.beta * self._cache[i] + (1.0 - self.beta) * g_sq
            denom = _vec_op(self._cache[i], lambda x: _safe_sqrt(x) + self.epsilon)
            update = self.lr * _vec_op2(g, denom, lambda a, b: a / b)
            params[i] = _apply_update(p, update)

        self.iterations += 1

    def reset(self):
        super().reset()
        self._cache = None

    def get_config(self):
        return {**super().get_config(), "beta": self.beta, "epsilon": self.epsilon}


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                epsilon: float = 1e-8):
        super().__init__(lr)
        if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m = None
        self._v = None

    def step(self, params: List[Union[float, Vector, Matrix]],
            grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._m is None:
            self._m = [self._zeros_like(p) for p in params]
            self._v = [self._zeros_like(p) for p in params]

        self.iterations += 1
        t = self.iterations
        # alpha_t already encodes bias correction
        alpha_t = self.lr * math.sqrt(1.0 - self.beta2 ** t) / (1.0 - self.beta1 ** t)

        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            g_sq = _vec_op(g, lambda x: x * x)
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g_sq
            denom = _vec_op(self._v[i], lambda x: _safe_sqrt(x) + self.epsilon)
            update = alpha_t * _vec_op2(self._m[i], denom, lambda a, b: a / b)
            params[i] = _apply_update(p, update)

    def reset(self):
        super().reset()
        self._m = None
        self._v = None

    def get_config(self):
        return {**super().get_config(), "beta1": self.beta1, "beta2": self.beta2,
                "epsilon": self.epsilon}
