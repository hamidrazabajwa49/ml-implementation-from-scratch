import math
from typing import List


def numerical_gradient(f, x: list, h: float = 1e-5) -> list:
    grad = []
    for i in range(len(x)):
        x_fwd = x[:]
        x_bwd = x[:]
        x_fwd[i] += h
        x_bwd[i] -= h
        grad.append((f(x_fwd) - f(x_bwd)) / (2.0 * h))
    return grad


def numerical_hessian(f, x: list, h: float = 1e-4) -> list:
    n = len(x)
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            x_pp = x[:]; x_pm = x[:]; x_mp = x[:]; x_mm = x[:]
            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h
            H[i][j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * h * h)
    return H


def _vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def _vec_sub(a, b):    
    return [ai - bi for ai, bi in zip(a, b)]

def _vec_scale(a, s):  
    return [ai * s for ai in a]

def _vec_dot(a, b):    
    return sum(ai * bi for ai, bi in zip(a, b))

def _vec_norm(a):      
    return math.sqrt(sum(ai * ai for ai in a))

def _mat_vec(M, v):    
    return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]


class ConvergenceTracker:

    def __init__(self, tol: float = 1e-6, patience: int = 10):
        self.tol = tol
        self.patience = patience
        self.history = []
        self._no_improve = 0
        self._best = float("inf")

    def update(self, loss: float) -> bool:
        self.history.append(loss)
        if loss < self._best - self.tol:
            self._best = loss
            self._no_improve = 0
        else:
            self._no_improve += 1
        return self._no_improve >= self.patience

    def converged(self) -> bool:
        if len(self.history) < 2:
            return False
        return abs(self.history[-1] - self.history[-2]) < self.tol

    def reset(self):
        self.history = []
        self._no_improve = 0
        self._best = float("inf")

    def plot_ascii(self, width: int = 60, height: int = 12):
        if not self.history:
            print("No history to plot.")
            return
        losses = self.history
        lo, hi = min(losses), max(losses)
        span = hi - lo if hi != lo else 1.0
        print(f"\n  Loss curve  (iter 0 → {len(losses)-1})")
        print(f"  max: {hi:.6f}")
        for row in range(height):
            threshold = hi - (row / (height - 1)) * span
            line = ""
            for val in losses:
                line += "█" if val >= threshold - span / height else " "
            print(f"  |{line}")
        print(f"  min: {lo:.6f}")
        print(f"  {'─' * len(losses)}")


def optimize(f,grad_f,x0: list,optimizer,max_iter: int = 1000,tol: float = 1e-6,verbose: bool = False,) -> dict:
    x = x0[:]
    tracker = ConvergenceTracker(tol=tol, patience=20)

    for i in range(max_iter):
        loss = f(x)
        grads = grad_f(x)
        tracker.update(loss)
        optimizer.record(loss)

        if verbose and i % max(1, max_iter // 10) == 0:
            gnorm = _vec_norm(grads)
            print(f"  iter {i:5d}  loss={loss:.8f}  |grad|={gnorm:.2e}")

        if _vec_norm(grads) < tol:
            if verbose:
                print(f"  converged at iter {i}  (grad norm < {tol})")
            break

        optimizer.step(x, grads)

    return {
        "x":          x,
        "loss":       f(x),
        "iterations": i + 1,
        "history":    optimizer.history[:],
        "converged":  _vec_norm(grad_f(x)) < tol * 100,
    }
