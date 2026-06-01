# Optimization Module

Built from scratch in pure Python — no NumPy, no SciPy.
Part of a 60-project ML/AI foundations series.

---

## Overview

Every machine learning model learns by minimizing a loss function. This module implements
the full optimization stack that makes that possible — from vanilla gradient descent to
adaptive second-order methods — all built from mathematical first principles.

Five files. One clean abstraction layer. Every optimizer shares the same interface so they
can be swapped in and out without changing any surrounding code.

```
optimization/
├── base.py          ← abstract Optimizer base class
├── first_order.py   ← GD, SGD, Momentum, RMSProp, Adam
├── second_order.py  ← Newton's Method, quasi-Newton (BFGS)
├── line_search.py   ← Backtracking, Golden Section
└── utils.py         ← numerical gradient, convergence tracker, history logger
```

---

## base.py — Abstract Optimizer

All optimizers inherit from a single base class that enforces a consistent interface.

```python
class Optimizer:
    def step(self, params, grads):
        """Update parameters given gradients. Returns updated params."""
        raise NotImplementedError

    def reset(self):
        """Reset internal state (moments, history, step count)."""
        raise NotImplementedError
```

Every optimizer implements `step()` and `reset()`. Nothing else is required externally.

---

## first_order.py — First-Order Methods

First-order methods use only gradient information — the direction of steepest descent.

---

### Gradient Descent (GD)

The foundation. Moves parameters in the direction opposite to the gradient.

```
θ ← θ − α · ∇f(θ)
```

| Parameter | Meaning              | Default |
|-----------|----------------------|---------|
| `lr`      | Learning rate α      | 0.01    |

```python
optimizer = GradientDescent(lr=0.01)
params = optimizer.step(params, grads)
```

**Limitation:** Uses the full dataset per step. Slow on large data. No memory of past gradients.

---

### Stochastic Gradient Descent (SGD)

Same update rule as GD but applied to a random mini-batch of data each step.
Introduces noise that helps escape shallow local minima.

```
θ ← θ − α · ∇f(θ; x_batch)
```

```python
optimizer = SGD(lr=0.01)
params = optimizer.step(params, grads_from_batch)
```

**When to use:** Large datasets. Early training. When computational cost per step matters.

---

### Momentum

Accumulates a velocity vector in directions of persistent gradient — dampens oscillation,
accelerates in consistent directions.

```
v ← β · v + α · ∇f(θ)
θ ← θ − v
```

| Parameter | Meaning                  | Default |
|-----------|--------------------------|---------|
| `lr`      | Learning rate α          | 0.01    |
| `beta`    | Momentum coefficient β   | 0.9     |

```python
optimizer = Momentum(lr=0.01, beta=0.9)
params = optimizer.step(params, grads)
```

**Intuition:** A ball rolling downhill gains speed. β controls how much of the previous
velocity is carried forward. β = 0 recovers plain SGD.

---

### RMSProp

Adapts the learning rate per parameter by dividing by a running average of squared
gradients. Prevents the learning rate from decaying too fast in directions with
consistently large gradients.

```
s ← ρ · s + (1 − ρ) · ∇f(θ)²
θ ← θ − (α / √(s + ε)) · ∇f(θ)
```

| Parameter | Meaning                        | Default |
|-----------|--------------------------------|---------|
| `lr`      | Global learning rate α         | 0.001   |
| `rho`     | Decay rate ρ                   | 0.9     |
| `eps`     | Numerical stability term ε     | 1e-8    |

```python
optimizer = RMSProp(lr=0.001, rho=0.9)
params = optimizer.step(params, grads)
```

**Intuition:** Parameters with noisy, large gradients get a smaller effective learning rate.
Parameters with small gradients get a larger one. Each parameter adapts independently.

---

### Adam (Adaptive Moment Estimation)

Combines Momentum (first moment) and RMSProp (second moment) with bias correction.
The most widely used optimizer in deep learning.

```
m ← β₁ · m + (1 − β₁) · ∇f(θ)          ← first moment  (direction)
v ← β₂ · v + (1 − β₂) · ∇f(θ)²         ← second moment (magnitude)

m̂ ← m / (1 − β₁ᵗ)                       ← bias correction
v̂ ← v / (1 − β₂ᵗ)                       ← bias correction

θ ← θ − α · m̂ / (√v̂ + ε)
```

| Parameter | Meaning                        | Default |
|-----------|--------------------------------|---------|
| `lr`      | Learning rate α                | 0.001   |
| `beta1`   | First moment decay β₁          | 0.9     |
| `beta2`   | Second moment decay β₂         | 0.999   |
| `eps`     | Numerical stability ε          | 1e-8    |

```python
optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
params = optimizer.step(params, grads)
```

**Why bias correction?** At step t=1, m and v are initialized to zero. Without correction,
early estimates are heavily biased toward zero. Dividing by (1 − βᵗ) corrects this — it
approaches 1.0 as t grows and the bias disappears.

**The secret:** Adam isn't magic. It tracks gradient direction (m̂) and gradient magnitude
(v̂) separately, then scales each parameter's update by both. That's why it works across
such a wide range of problems without manual tuning.

---

## second_order.py — Second-Order Methods

Second-order methods use curvature information — the Hessian — to take more informed
steps. Faster convergence per iteration, but expensive to compute.

---

### Newton's Method

Uses the exact Hessian (matrix of second derivatives) to scale the gradient update.
Corrects for the curvature of the loss surface.

```
θ ← θ − H⁻¹ · ∇f(θ)
```

Where H is the Hessian matrix: `Hᵢⱼ = ∂²f / ∂θᵢ ∂θⱼ`

```python
optimizer = NewtonMethod()
params = optimizer.step(params, grads, hessian)
```

**Limitation:** Computing and inverting H costs O(n³) for n parameters. Impractical
beyond small problems. This motivates quasi-Newton methods.

**Convergence:** Quadratic near the optimum — the number of correct digits roughly
doubles each iteration. Far faster than gradient descent close to the solution.

---

### BFGS (Broyden–Fletcher–Goldfarb–Shanno)

A quasi-Newton method. Approximates the inverse Hessian H⁻¹ iteratively using only
gradient evaluations — never computes H directly.

```
sₖ = θₖ₊₁ − θₖ               ← parameter change
yₖ = ∇f(θₖ₊₁) − ∇f(θₖ)      ← gradient change

Hₖ₊₁⁻¹ updated via rank-2 correction (Sherman-Morrison-Woodbury formula)

θ ← θ − Hₖ⁻¹ · ∇f(θ)
```

```python
optimizer = BFGS()
params = optimizer.step(params, grads)   # H⁻¹ updated internally
```

**Intuition:** Each step uses the history of parameter and gradient changes to build a
progressively better approximation of the curvature. After enough steps, the
approximation becomes accurate enough to enable Newton-like convergence.

**Cost:** O(n²) per step instead of O(n³) for exact Newton. Still scales poorly to
very high dimensions — which is why deep learning uses first-order methods.

---

## line_search.py — Line Search

Line search answers the question first-order methods ignore: *how far should we step?*
Given a direction p, find the step size α that sufficiently decreases f.

---

### Backtracking Line Search

Starts with a large step size and shrinks it until the Armijo sufficient decrease
condition is satisfied.

```
f(θ + α · p) ≤ f(θ) + c · α · ∇f(θ)ᵀ · p
```

| Parameter | Meaning                         | Default |
|-----------|---------------------------------|---------|
| `alpha`   | Initial step size               | 1.0     |
| `rho`     | Shrink factor                   | 0.5     |
| `c`       | Sufficient decrease constant    | 1e-4    |

```python
alpha = backtracking(f, theta, grad, direction, alpha=1.0, rho=0.5, c=1e-4)
```

**Intuition:** The Armijo condition ensures we make meaningful progress — not just a
tiny improvement that barely justifies the step. If the condition fails, halve α and
try again.

---

### Golden Section Search

Finds the minimum of a unimodal function over a bounded interval [a, b] without
requiring derivatives. Brackets the minimum by evaluating at two interior points
spaced by the golden ratio φ = (√5 − 1) / 2 ≈ 0.618.

```
φ = (√5 − 1) / 2  ≈  0.618

x₁ = b − φ · (b − a)
x₂ = a + φ · (b − a)

Discard the worse half. Repeat.
```

```python
alpha_min = golden_section(f_along_direction, a=0.0, b=1.0, tol=1e-5)
```

**Convergence:** Reduces the interval by factor φ ≈ 0.618 per iteration.
Reaches tolerance in O(log(1/tol)) evaluations — no gradients needed.

---

## utils.py — Utilities

### Numerical Gradient

Approximates gradients using finite differences. Used to verify analytic gradient
implementations (gradient checking).

```
∂f/∂θᵢ ≈ [f(θ + εeᵢ) − f(θ − εeᵢ)] / (2ε)
```

```python
grad_numerical = numerical_gradient(f, params, eps=1e-5)
grad_analytic  = your_grad_function(params)

# Should be close to zero for a correct implementation
error = max(abs(grad_numerical - grad_analytic))
```

**When to use:** After implementing a new optimizer or loss function, run gradient
checking before trusting the analytic gradients.

---

### Convergence Tracker

Monitors whether optimization has converged based on gradient norm and parameter change.

```python
tracker = ConvergenceTracker(tol=1e-6, patience=10)

for step in range(max_steps):
    grads = compute_grads(params)
    params = optimizer.step(params, grads)

    if tracker.check(params, grads):
        print(f"Converged at step {step}")
        break
```

Convergence is declared when `‖∇f(θ)‖ < tol` holds for `patience` consecutive steps.

---

### History Logger

Records loss, gradient norm, and parameter values at each step for post-training analysis.

```python
history = HistoryLogger()
history.log(step=t, loss=loss_val, grad_norm=norm, params=params)

history.plot_loss()       # loss curve
history.plot_grad_norm()  # gradient norm over time
history.summary()         # min loss, convergence step, total steps
```

---

## Optimizer Comparison

| Optimizer     | Memory  | Per-step Cost | Adapts LR | Convergence         |
|---------------|---------|---------------|-----------|---------------------|
| GD            | O(n)    | O(n)          | No        | Linear              |
| SGD           | O(n)    | O(n)          | No        | Noisy / linear      |
| Momentum      | O(n)    | O(n)          | No        | Faster than SGD     |
| RMSProp       | O(n)    | O(n)          | Yes       | Good for RNNs       |
| Adam          | O(n)    | O(n)          | Yes       | Fast, general       |
| Newton        | O(n²)   | O(n³)         | Yes       | Quadratic (local)   |
| BFGS          | O(n²)   | O(n²)         | Yes       | Super-linear        |

---

## Mathematical Identities (verified by tests)

```
Adam with β₁=0, β₂=0    →  recovers SGD with lr scaling
Momentum with β=0        →  recovers GD exactly
BFGS step 0              →  equivalent to gradient descent (H⁻¹ = I)
Numerical gradient error  =  O(ε²) for central differences
Golden section reduction  =  φ ≈ 0.618 per iteration
Armijo condition c=0      →  any decrease accepted (too loose)
Armijo condition c=1      →  exact line minimization (too strict)
```

---

## Used in Later Projects

| Project | Component Used |
|---|---|
| Linear Regression (#11) | `GradientDescent`, `ConvergenceTracker` |
| Logistic Regression (#12) | `SGD`, `Adam`, `HistoryLogger` |
| Neural Network MLP (#20) | `Adam`, `Momentum`, `numerical_gradient` |
| CNN from scratch (#21) | `Adam`, `HistoryLogger` |
| Minigrad autograd engine (#25) | `GradientDescent`, `backtracking` |
| Transformer from scratch (#54) | `Adam` with warmup scheduling |
| LoRA fine-tuning (#60) | `Adam`, `convergence_tracker` |

---

## Quick Start

```python
from optimization.first_order import Adam
from optimization.utils import HistoryLogger, ConvergenceTracker, numerical_gradient

# Define a simple quadratic loss
def loss(params):
    x, y = params
    return (x - 3.0) ** 2 + (y + 1.0) ** 2

def grads(params):
    x, y = params
    return [2 * (x - 3.0), 2 * (y + 1.0)]

# Initialize
params = [0.0, 0.0]
optimizer = Adam(lr=0.1)
tracker = ConvergenceTracker(tol=1e-6, patience=5)
history = HistoryLogger()

# Optimize
for step in range(1000):
    g = grads(params)
    params = optimizer.step(params, g)
    history.log(step, loss(params), sum(gi**2 for gi in g)**0.5, params)

    if tracker.check(params, g):
        print(f"Converged at step {step}. Loss: {loss(params):.8f}")
        break

# Verify gradients
print(numerical_gradient(loss, params))   # should match grads(params)
```

---

## Dependencies

```
Python 3.10+  (standard library: math only)
```

No external packages.

---

## Part of

**60-Day ML/AI Foundations Challenge**
Building every algorithm from scratch — linear algebra → statistics → ML → deep learning → LLMs.

**Phase 1 — Math Foundations (10/10 complete)**

| # | Module |
|---|--------|
| 1 | Vector operations library |
| 2 | Matrix operations library |
| 3 | Eigenvalues & eigenvectors |
| 4 | SVD module |
| 5 | Probability distributions |
| 6 | Descriptive statistics |
| 7 | Hypothesis testing |
| 8 | Bayesian inference |
| 9 | Information theory |
| **10** | **Optimization ← this module** |

→ [GitHub](https://github.com/hamidrazabajwa49) | [LinkedIn](https://linkedin.com/in/hamid-raza-bajwa-564a91377)
