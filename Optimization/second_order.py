import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Optimization.base import Optimizer
from Optimization.utils import numerical_hessian, _vec_add, _vec_sub, _vec_scale, _vec_dot, _mat_vec


def _solve_linear(A: list, b: list) -> list:
    n = len(b)
    if n == 0:
        return []
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if abs(M[row][col]) > 1e-14:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Singular system: matrix is not invertible")
        M[col], M[pivot] = M[pivot], M[col]
        scale = M[col][col]
        M[col] = [v / scale for v in M[col]]
        for row in range(n):
            if row != col:
                factor = M[row][col]
                M[row] = [M[row][k] - factor * M[col][k] for k in range(n + 1)]

    return [M[i][n] for i in range(n)]


def _identity(n: int) -> list:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_mat(A: list, B: list) -> list:
    n = len(A)
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]


def _outer(u: list, v: list) -> list:
    return [[ui * vj for vj in v] for ui in u]


def _mat_scale(M: list, s: float) -> list:
    return [[Mij * s for Mij in row] for row in M]


def _mat_add(A: list, B: list) -> list:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


class NewtonMethod(Optimizer):
    def __init__(self, f, lr: float = 1.0, hessian_h: float = 1e-4):
        super().__init__(lr)
        if hessian_h <= 0.0:
            raise ValueError(f"hessian_h must be positive, got {hessian_h}")
        self.f = f
        self.hessian_h = hessian_h

    def step(self, params: list, grads: list) -> list:
        H = numerical_hessian(self.f, params, h=self.hessian_h)
        try:
            direction = _solve_linear(H, grads)
        except ValueError:
            direction = grads[:]
        new_params = [p - self.lr * d for p, d in zip(params, direction)]
        self.iterations += 1
        return new_params

    def step_with_hessian(self, params: list, grads: list, hessian: list) -> list:
        try:
            direction = _solve_linear(hessian, grads)
        except ValueError:
            direction = grads[:]
        new_params = [p - self.lr * d for p, d in zip(params, direction)]
        self.iterations += 1
        return new_params


class BFGS(Optimizer):
    def __init__(self, lr: float = 1.0, eps: float = 1e-10):
        super().__init__(lr)
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps
        self._H_inv = None
        self._x_prev = None
        self._g_prev = None

    def step(self, params: list, grads: list) -> list:
        n = len(params)

        if self._H_inv is None:
            self._H_inv = _identity(n)

        # Compute search direction using current H_inv
        direction = _vec_scale(_mat_vec(self._H_inv, grads), -1.0)

        # Take step
        x_new = _vec_add(params, _vec_scale(direction, self.lr))

        # Clamp step size to avoid divergence
        step = _vec_sub(x_new, params)
        step_norm = sum(s ** 2 for s in step) ** 0.5
        max_step = 10.0
        if step_norm > max_step:
            x_new = _vec_add(params, _vec_scale(step, max_step / step_norm))

        # Update H_inv using curvature from PREVIOUS step
        # (s = x_current - x_prev, y = g_current - g_prev)
        if self._x_prev is not None:
            s = _vec_sub(params, self._x_prev)
            y = _vec_sub(grads, self._g_prev)
            sy = _vec_dot(s, y)

            if abs(sy) > self.eps:
                Hy = _mat_vec(self._H_inv, y)
                yHy = _vec_dot(y, Hy)
                rho = 1.0 / sy

                term1 = _mat_scale(_outer(s, s), rho * (1.0 + rho * yHy))
                term2 = _mat_scale(_outer(s, Hy), rho)
                term3 = _mat_scale(_outer(Hy, s), rho)

                self._H_inv = _mat_add(
                    _mat_add(self._H_inv, term1),
                    _mat_add(
                        [[-v for v in row] for row in term2],
                        [[-v for v in row] for row in term3],
                    ),
                )

        self._x_prev = params[:]
        self._g_prev = grads[:]
        self.iterations += 1
        return x_new

    def reset(self):
        super().reset()
        self._H_inv = None
        self._x_prev = None
        self._g_prev = None
