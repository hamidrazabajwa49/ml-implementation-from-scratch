import os
import sys
import math
from base import Optimizer
from typing import List, Union


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Vectors.vector import Vector
from Matrix.matrix import Matrix


class GradientDescent(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self, params: List[Union[float, Vector, Matrix]],
            grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        for i, (p, g) in enumerate(zip(params, grads)):
            params[i] = p - self.lr * g
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

    def step(self, params: List[Union[float, Vector, Matrix]],grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")

        lr_eff = self.lr / (1.0 + self.decay * self.iterations) if self.decay > 0 else self.lr

        if self._velocity is None:
            self._velocity = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.momentum > 0:
                self._velocity[i] = self.momentum * self._velocity[i] + g
                params[i] = p - lr_eff * self._velocity[i]
            else:
                params[i] = p - lr_eff * g

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

    def step(self, params: List[Union[float, Vector, Matrix]],grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._velocity is None:
            self._velocity = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            self._velocity[i] = self.beta * self._velocity[i] + (1.0 - self.beta) * g
            params[i] = p - self.lr * self._velocity[i]

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
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.beta = beta
        self.epsilon = epsilon
        self._cache = None

    def step(self, params: List[Union[float, Vector, Matrix]],grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._cache is None:
            self._cache = [self._zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            g_sq = g.element_wise(lambda x: x * x) if hasattr(g, 'element_wise') else (g * g)
            self._cache[i] = self.beta * self._cache[i] + (1.0 - self.beta) * g_sq

            denom = self._cache[i].element_wise(lambda x: math.sqrt(x) + self.epsilon) if hasattr(self._cache[i], 'element_wise') else (math.sqrt(self._cache[i]) + self.epsilon)

            update = g.element_wise_with(denom, lambda a, b: a / b) * self.lr if hasattr(g, 'element_wise_with') else (g / denom * self.lr)

            params[i] = p - update

        self.iterations += 1

    def reset(self):
        super().reset()
        self._cache = None

    def get_config(self):
        return {**super().get_config(), "beta": self.beta, "epsilon": self.epsilon}


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(lr)
        if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m = None
        self._v = None

    def step(self, params: List[Union[float, Vector, Matrix]],grads: List[Union[float, Vector, Matrix]]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        if self._m is None:
            self._m = [self._zeros_like(p) for p in params]
            self._v = [self._zeros_like(p) for p in params]

        self.iterations += 1
        t = self.iterations
        alpha_t = self.lr * math.sqrt(1.0 - self.beta2 ** t) / (1.0 - self.beta1 ** t)

        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            g_sq = g.element_wise(lambda x: x * x) if hasattr(g, 'element_wise') else (g * g)
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g_sq

            m_hat = self._m[i] / (1.0 - self.beta1 ** t)
            v_hat = self._v[i] / (1.0 - self.beta2 ** t)

            denom = v_hat.element_wise(lambda x: math.sqrt(x) + self.epsilon) if hasattr(v_hat, 'element_wise') else (math.sqrt(v_hat) + self.epsilon)
            update = m_hat.element_wise_with(denom, lambda a, b: a / b) * alpha_t if hasattr(m_hat, 'element_wise_with') else (m_hat / denom * alpha_t)

            params[i] = p - update

    def reset(self):
        super().reset()
        self._m = None
        self._v = None

    def get_config(self):
        return {**super().get_config(), "beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon}

  
