# Optimization — Pure Python Optimization Library

## Overview

The `Optimization/` package is a pure Python implementation of gradient-based optimization algorithms — built from scratch as part of the *Engineering Redemption Arc*, a structured 60-project ML engineering curriculum in the [`ml-implementation-from-scratch`](https://github.com/) repository.

The package covers first-order optimizers (GD, SGD with momentum and decay, Momentum, RMSProp, Adam), second-order methods (Newton's Method, BFGS), line search strategies (backtracking, golden section), numerical differentiation utilities, a convergence tracker, and a unified `optimize()` loop — all without NumPy or SciPy.

Parameters can be scalars (`float`), `Vector` objects, or `Matrix` objects from the sibling packages, enabling the same optimizers to drive both toy functions and full model weight updates.

---

## Project Structure

```
ml-implementation-from-scratch/
├── Vectors/
│   └── vector.py                  # Vector primitive
├── Matrix/
│   └── matrix.py                  # Matrix primitive
└── Optimization/
    ├── base.py                    # Optimizer abstract base class
    ├── first_order.py             # GradientDescent, SGD, Momentum, RMSProp, Adam
    ├── second_order.py            # NewtonMethod, BFGS
    ├── line_search.py             # backtracking(), golden_section()
    └── utils.py                   # Numerical differentiation, ConvergenceTracker, optimize()
```

---

## Dependencies

| Requirement | Detail |
|---|---|
| Python | 3.10+ |
| `math`, `os`, `sys`, `typing` | Standard library only |
| `Vectors/vector.py` | Parameter type support in `base.py` and `first_order.py` |
| `Matrix/matrix.py` | Parameter type support in `base.py` and `first_order.py` |

`line_search.py` is fully self-contained — it has no local imports.

No `pip install` required.

---

## Installation

```python
import sys
sys.path.insert(0, "/path/to/ml-implementation-from-scratch")

from Optimization.first_order import GradientDescent, SGD, Momentum, RMSProp, Adam
from Optimization.second_order import NewtonMethod, BFGS
from Optimization.line_search import backtracking, golden_section
from Optimization.utils import numerical_gradient, numerical_hessian, optimize, ConvergenceTracker
```

---

## Module Reference

---

### `base.py` — Abstract Base Class

#### `Optimizer(lr=0.01)`

All optimizers inherit from this class. Provides shared state and helpers.

```python
optimizer.lr             # learning rate
optimizer.iterations     # step count
optimizer.history        # list of recorded losses

optimizer.step(params, grads)   # abstract — implemented by subclasses
optimizer.reset()               # resets iterations, history, and any internal state
optimizer.record(loss)          # appends a loss value to history
optimizer.get_config()          # returns {"lr": ...}
optimizer._zeros_like(x)        # returns a zero of the same type as x (float, Vector, Matrix)
```

`lr` must be positive. `step()` raises `NotImplementedError` if called on `Optimizer` directly.

---

### `first_order.py` — First-Order Optimizers

All first-order optimizers accept `params` and `grads` as lists whose elements can be `float`, `Vector`, or `Matrix`. They mutate `params` in place and increment `self.iterations`.

---

#### `GradientDescent(lr=0.01)`

Vanilla gradient descent: `p ← p − lr · g`.

```python
from Optimization.first_order import GradientDescent

opt = GradientDescent(lr=0.1)
params = [2.0, -1.0]
grads  = [0.4,  0.8]
opt.step(params, grads)
# params is now updated in place
```

---

#### `SGD(lr=0.01, momentum=0.0, decay=0.0)`

SGD with optional momentum and learning rate decay.

```python
from Optimization.first_order import SGD

opt = SGD(lr=0.01, momentum=0.9, decay=1e-4)
opt.step(params, grads)
```

- **Momentum:** `v ← momentum·v + lr_eff·g`, then `p ← p − v`.
- **Decay:** `lr_eff = lr / (1 + decay × t)` where `t` is the iteration count.
- When `momentum=0.0` (default), reduces to plain SGD with optional decay.
- `momentum` must be in `[0, 1)`. `decay` must be non-negative.

---

#### `Momentum(lr=0.01, beta=0.9)`

Exponential moving average of gradients: `v ← β·v + (1−β)·g`, then `p ← p − lr·v`.

```python
from Optimization.first_order import Momentum

opt = Momentum(lr=0.01, beta=0.9)
opt.step(params, grads)
```

`beta` must be in `[0, 1)`. Velocity is initialized to zeros on the first `step()` call.

> **Distinction from SGD with momentum:** The `SGD` class uses the standard formulation `v ← β·v + lr·g`. `Momentum` uses the EMA formulation `v ← β·v + (1−β)·g`, which keeps the scale of `v` closer to that of `g`.

---

#### `RMSProp(lr=0.001, beta=0.9, epsilon=1e-8)`

Adapts the learning rate per parameter using an EMA of squared gradients.

```python
from Optimization.first_order import RMSProp

opt = RMSProp(lr=0.001, beta=0.9, epsilon=1e-8)
opt.step(params, grads)
```

Update rule: `cache ← β·cache + (1−β)·g²`, then `p ← p − lr · g / (√cache + ε)`.

`beta` must be in `[0, 1)`. `epsilon` must be positive.

---

#### `Adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)`

Adaptive moment estimation — maintains first and second moment estimates with bias correction.

```python
from Optimization.first_order import Adam

opt = Adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
opt.step(params, grads)
```

Bias correction is folded into `alpha_t = lr · √(1−β2ᵗ) / (1−β1ᵗ)`, computed once per step. Both `beta1` and `beta2` must be in `[0, 1)`. `epsilon` must be positive.

---

### `second_order.py` — Second-Order Optimizers

Second-order optimizers operate on plain Python `list` parameters (not `Vector` or `Matrix`). Their `step()` methods **return** the new parameter list rather than mutating in place.

---

#### `NewtonMethod(f, lr=1.0, hessian_h=1e-4)`

Uses the numerical Hessian to compute the Newton direction `H⁻¹g`, then takes a step `p ← p − lr · H⁻¹g`.

```python
from Optimization.second_order import NewtonMethod

def f(x):
    return x[0]**2 + 2*x[1]**2

opt = NewtonMethod(f=f, lr=1.0)
params = [3.0, -2.0]
grads  = [6.0, -8.0]

params = opt.step(params, grads)   # returns new params
```

The Hessian is computed via `numerical_hessian()` from `utils.py`. If the system is singular, the method falls back to the gradient direction. For a known analytical Hessian, use `step_with_hessian(params, grads, hessian)` to skip the numerical computation.

`hessian_h` is the finite difference step size (default `1e-4`).

---

#### `BFGS(lr=1.0, eps=1e-10)`

Quasi-Newton method that approximates the inverse Hessian `H⁻¹` iteratively using rank-2 updates from observed curvature.

```python
from Optimization.second_order import BFGS

opt = BFGS(lr=1.0)
params = opt.step(params, grads)   # first step uses identity H_inv
params = opt.step(params, grads)   # subsequent steps use accumulated curvature
```

- Initialized with `H_inv = I`.
- The inverse Hessian update uses the BFGS formula: `H⁻¹ ← H⁻¹ + ρ(1 + ρ·yᵀH⁻¹y)·ssᵀ − ρ(sHyᵀ + Hysᵀ)` where `s = xₜ − xₜ₋₁`, `y = gₜ − gₜ₋₁`, `ρ = 1/(sᵀy)`.
- Updates are skipped when `|sᵀy| < eps` (curvature condition not met).
- Step size is clamped to a maximum norm of `10.0` to prevent divergence.
- `reset()` clears `H_inv`, `x_prev`, and `g_prev`.

---

### `line_search.py` — Line Search Methods

Standalone functions with no local imports. Operate on plain Python `list` or `float` parameters.

---

#### `backtracking(f, x, grad, direction, alpha=1.0, rho=0.5, c=1e-4, max_iter=100)`

Armijo backtracking line search. Reduces the step size `alpha` by factor `rho` until the Armijo sufficient decrease condition is satisfied.

```python
from Optimization.line_search import backtracking

alpha = backtracking(
    f=objective,
    x=[1.0, 2.0],
    grad=[0.4, 0.8],
    direction=[-0.4, -0.8],  # must be a descent direction (slope < 0)
    alpha=1.0,
    rho=0.5,
    c=1e-4
)
# returns float: the accepted step size
```

Condition: `f(x + α·d) ≤ f(x) + c·α·(∇f·d)`.

Raises `ValueError` if `direction` is not a descent direction (i.e., `∇f · d ≥ 0`). Returns the last `alpha` if `max_iter` is exhausted without satisfying the condition.

| Parameter | Meaning | Constraint |
|---|---|---|
| `alpha` | Initial step size | `> 0` |
| `rho` | Reduction factor | `(0, 1)` |
| `c` | Sufficient decrease constant | `(0, 1)` |

---

#### `golden_section(f, a, b, tol=1e-8, max_iter=200)`

Finds the minimum of a unimodal scalar function on `[a, b]` using the golden-section search.

```python
from Optimization.line_search import golden_section

result = golden_section(f=lambda x: (x - 2.3)**2, a=0.0, b=5.0)
# {
#     "x_min": float,
#     "f_min": float,
#     "bracket": (a, b),      # final bracket
#     "converged": bool
# }
```

`a` must be strictly less than `b`. Terminates when `|b − a| < tol` or `max_iter` is reached.

---

### `utils.py` — Numerical Utilities

---

#### `numerical_gradient(f, x, h=1e-5)`

Central difference gradient: `∂f/∂xᵢ ≈ (f(x+heᵢ) − f(x−heᵢ)) / 2h`.

```python
from Optimization.utils import numerical_gradient

grad = numerical_gradient(f=lambda x: x[0]**2 + x[1]**2, x=[1.0, 2.0])
# [2.0, 4.0]
```

`h` must be positive. Returns a `list` of the same length as `x`.

---

#### `numerical_hessian(f, x, h=1e-4)`

Second-order mixed partial derivatives via four-point finite differences:
`∂²f/∂xᵢ∂xⱼ ≈ (f(x+heᵢ+heⱼ) − f(x+heᵢ−heⱼ) − f(x−heᵢ+heⱼ) + f(x−heᵢ−heⱼ)) / 4h²`.

```python
from Optimization.utils import numerical_hessian

H = numerical_hessian(f=lambda x: x[0]**2 + 2*x[1]**2, x=[1.0, 1.0])
# [[2.0, 0.0], [0.0, 4.0]]
```

Returns an `n×n` `list[list[float]]`. Used internally by `NewtonMethod`.

---

#### `ConvergenceTracker(tol=1e-6, patience=10)`

Monitors training loss for convergence and early stopping.

```python
from Optimization.utils import ConvergenceTracker

tracker = ConvergenceTracker(tol=1e-6, patience=10)

for i in range(max_iter):
    loss = compute_loss()
    should_stop = tracker.update(loss)   # returns True when patience is exhausted
    if should_stop:
        break

tracker.converged()        # True if last two losses differ by < tol
tracker.history            # list of all recorded losses
tracker.plot_ascii()       # prints an ASCII loss curve to stdout
tracker.reset()            # clears all state
```

`update()` returns `True` when loss has not improved by more than `tol` for `patience` consecutive steps. `converged()` checks only the last two steps.

`plot_ascii(width=60, height=12)` renders a block-character loss curve in the terminal. Width scales automatically to the number of iterations.

---

#### `optimize(f, grad_f, x0, optimizer, max_iter=1000, tol=1e-6, verbose=False)`

Unified optimization loop compatible with all optimizers in the package.

```python
from Optimization.utils import optimize
from Optimization.first_order import Adam

def f(x):
    return (x[0] - 1)**2 + (x[1] + 2)**2

def grad_f(x):
    return [2*(x[0]-1), 2*(x[1]+2)]

opt = Adam(lr=0.01)
result = optimize(f, grad_f, x0=[0.0, 0.0], optimizer=opt, max_iter=500, verbose=True)
# {
#     "x": list,            # final parameters
#     "loss": float,        # f(x) at termination
#     "iterations": int,
#     "history": list,      # loss at each recorded step
#     "converged": bool     # True if final gradient norm < tol
# }
```

Convergence is declared when `‖∇f(x)‖ < tol`. The loop also uses `ConvergenceTracker` with `patience=20` to detect stagnation. First-order optimizers mutate `params` in place; second-order optimizers return new params — both are handled transparently.

---

## Example Session

```python
import math
from Optimization.first_order import Adam, SGD, RMSProp
from Optimization.second_order import NewtonMethod, BFGS
from Optimization.line_search import backtracking, golden_section
from Optimization.utils import numerical_gradient, optimize, ConvergenceTracker

# --- Rosenbrock function ---
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dx1 = 200*(x[1] - x[0]**2)
    return [dx0, dx1]

# Adam
result = optimize(rosenbrock, rosenbrock_grad, x0=[-1.0, 1.0],
                  optimizer=Adam(lr=0.01), max_iter=5000, verbose=True)
print(result["x"], result["converged"])

# SGD with momentum and decay
result = optimize(rosenbrock, rosenbrock_grad, x0=[-1.0, 1.0],
                  optimizer=SGD(lr=0.001, momentum=0.9, decay=1e-4), max_iter=5000)
print(result["loss"])

# Newton's Method
opt = NewtonMethod(f=rosenbrock, lr=1.0)
result = optimize(rosenbrock, rosenbrock_grad, x0=[0.5, 0.5], optimizer=opt, max_iter=50)
print(result["x"])

# BFGS
result = optimize(rosenbrock, rosenbrock_grad, x0=[0.0, 0.0],
                  optimizer=BFGS(lr=1.0), max_iter=200)
print(result["x"])

# Backtracking line search
alpha = backtracking(rosenbrock, x=[0.0, 0.0], grad=rosenbrock_grad([0.0, 0.0]),
                     direction=[-g for g in rosenbrock_grad([0.0, 0.0])])
print(f"Step size: {alpha:.6f}")

# Golden section on a scalar function
res = golden_section(lambda x: (x - 2.5)**2 + 1, a=0.0, b=5.0)
print(res)

# Numerical gradient check
grad_numerical = numerical_gradient(rosenbrock, [1.0, 1.0])
grad_analytical = rosenbrock_grad([1.0, 1.0])
print(grad_numerical, grad_analytical)

# Convergence tracker
tracker = ConvergenceTracker(tol=1e-5, patience=5)
tracker.update(1.0)
tracker.update(0.5)
tracker.update(0.5)
tracker.plot_ascii()
```

---

## Error Reference

| Situation | Exception |
|---|---|
| `lr <= 0` | `ValueError` |
| `momentum` outside `[0, 1)` | `ValueError` |
| `decay < 0` in SGD | `ValueError` |
| `beta`, `beta1`, `beta2` outside `[0, 1)` | `ValueError` |
| `epsilon <= 0` | `ValueError` |
| `hessian_h <= 0` in NewtonMethod | `ValueError` |
| `eps <= 0` in BFGS | `ValueError` |
| Mismatched `params` and `grads` lengths | `ValueError` |
| Singular Hessian in NewtonMethod | Falls back to gradient direction (no exception) |
| Non-descent direction in `backtracking` | `ValueError` |
| `alpha <= 0`, `rho` or `c` outside `(0,1)` in `backtracking` | `ValueError` |
| `a >= b` in `golden_section` | `ValueError` |
| `tol <= 0` or `patience < 1` in `ConvergenceTracker` | `ValueError` |
| `max_iter < 1` or `tol <= 0` in `optimize` | `ValueError` |
| `h <= 0` in `numerical_gradient`/`numerical_hessian` | `ValueError` |
| Unsupported parameter type in `_zeros_like` | `TypeError` |

---

## Design Notes

- **Polymorphic parameters:** First-order optimizers work with `float`, `Vector`, and `Matrix` parameters in the same `step()` call. The `_vec_op` and `_vec_op2` helpers in `first_order.py` apply element-wise lambdas across both scalar and `Vector` types. `Matrix` support is inherited through `Vector`'s arithmetic operators.
- **In-place vs. return semantics:** First-order optimizers mutate `params[i]` in place (they reassign list elements). Second-order optimizers return a new list. The `optimize()` loop handles both: `result = optimizer.step(x, grads); if result is not None: x = result`.
- **Lazy state initialization:** All optimizers initialize velocity, cache, and moment buffers to `None` and allocate zeros on the first `step()` call. This avoids requiring the parameter shape at construction time.
- **Adam bias correction:** Rather than maintaining `m_hat` and `v_hat` as separate variables, bias correction is folded into a single corrected learning rate `alpha_t` computed once per iteration.
- **BFGS curvature check:** The inverse Hessian update is skipped when `|sᵀy| < eps`. This guards against indefinite updates when the curvature condition is violated (e.g., non-convex regions).
- **`line_search.py` has no local imports:** It operates entirely on plain Python lists and floats, making it usable independently of the rest of the package.
- **`numerical_hessian` requires 4n² function evaluations:** It is exact up to O(h²) error but expensive for high-dimensional problems. Use analytical Hessians via `step_with_hessian()` in performance-critical settings.

---

## Roadmap Context

This package is **Project 10** of 60 in the Engineering Redemption Arc curriculum. It depends on:

- **`Vectors/vector.py`** — Project 1
- **`Matrix/matrix.py`** — Project 2

It underpins every model implementation in Phase 2 (Projects 11–25):

- **Linear and logistic regression** — gradient descent and Adam are the primary training loops.
- **Neural networks** — SGD with momentum, RMSProp, and Adam are the standard optimizers.
- **Second-order methods** — Newton and BFGS are used where curvature information accelerates convergence (small-scale problems, quasi-Newton fine-tuning).
- **`ConvergenceTracker` and `optimize()`** — provide the standardized training loop reused across all subsequent model implementations.
