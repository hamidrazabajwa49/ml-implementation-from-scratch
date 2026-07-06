# Optimization — Gradient-Based & Quasi-Newton Optimizers (From Scratch)

**Phase 1: Math Foundations**.

A dependency-free, pure-Python optimization library: first-order
optimizers (SGD family, RMSProp, Adam, AdaGrad) that work over
`float`/`Vector`/`Matrix` parameters, second-order optimizers (Newton's
method, BFGS) over flat float lists, 1D line search routines, and the
finite-difference/driver-loop utilities that tie them together. SciPy is
used only in the test suite, as a correctness oracle.

---

## Overview

| File | Provides |
|---|---|
| `base.py` | `Optimizer` — the shared base class and calling-convention contract |
| `first_order.py` | `GradientDescent`, `SGD`, `Momentum`, `AdaGrad`, `RMSProp`, `Adam` |
| `second_order.py` | `NewtonMethod`, `BFGS` + lightweight list-of-lists linear algebra helpers |
| `line_search.py` | `backtracking` (Armijo), `golden_section` |
| `utils.py` | `numerical_gradient`, `numerical_hessian`, flat-vector helpers, `ConvergenceTracker`, `optimize()` driver loop |

**Two calling conventions coexist by design** (documented in full in
`base.py`'s module docstring):

- **First-order** optimizers accept `params` as a list of
  `float`/`Vector`/`Matrix` elements and mutate it **in place**,
  returning `None`.
- **Second-order** optimizers operate on a flat list of floats and
  **return a new list** instead, since they need the whole vector at
  once to apply a (inverse) Hessian.

`utils.optimize()` drives either family transparently:
`result = optimizer.step(x, grads); if result is not None: x = result`.

---

## Project Structure

```
math_foundations/
└── Optimization/
    ├── base.py
    ├── first_order.py
    ├── second_order.py
    ├── line_search.py
    ├── utils.py
    ├── tests/
    │   ├── test_base.py
    │   ├── test_first_order.py
    │   ├── test_second_order.py
    │   ├── test_line_search.py
    │   └── test_utils.py
    └── README.md
```

`base.py` and `first_order.py` import `Vector`/`Matrix` from the sibling
`Vectors`/`Matrix` modules; `second_order.py` imports from `base.py` and
`utils.py`. `line_search.py` and `utils.py` (aside from the base-optimizer
type hint) are otherwise self-contained. `Vectors`, `Matrix`, and
`Optimization` must all live side-by-side under `math_foundations/`.

---

## Dependencies

| Component | Requires |
|---|---|
| Library files | Python 3.8+ standard library only (`math`, `logging`, `typing`) + `Vectors.vector.Vector` / `Matrix.matrix.Matrix` (sibling modules, for `base.py`/`first_order.py`) |
| `tests/*.py` | `pytest`, `scipy` (test-only, for regression checks against `scipy.optimize`) |

Install test dependencies:

```bash
pip install pytest scipy
```

Run the full suite from `math_foundations/Optimization/`:

```bash
pytest tests/ -v
```

---

## Module Reference

### `base.py`

```python
class Optimizer:
    def __init__(self, lr: float = 0.01)
    def step(self, params, grads) -> None | List[float]     # abstract; raises NotImplementedError
    def reset(self) -> None                                    # clears iterations/history + subclass state
    def get_config(self) -> Dict[str, float]                   # {"lr": ...}
    def record(self, loss: float) -> None                      # appends to self.history
    def _zeros_like(self, x) -> float | Vector | Matrix
```
Shared base class. `lr` is validated positive/finite/non-NaN/non-bool at construction. `_zeros_like` returns a zero-valued object matching `x`'s type (`float`/`Vector`/`Matrix`) — used by stateful optimizers to initialize per-parameter accumulators (velocity, squared-gradient cache); raises `TypeError` for `bool` or any other unsupported type.

### `first_order.py`

All classes below accept `params`/`grads` as lists of `float`/`Vector`/`Matrix` (one entry per parameter *tensor*), mutate `params` **in place**, and return `None`. All raise `ValueError` if `len(params) != len(grads)`.

| Class | Update rule | Notes |
|---|---|---|
| `GradientDescent(lr=0.01)` | `param -= lr * grad` | Vanilla batch gradient descent |
| `SGD(lr, momentum=0, decay=0, nesterov=False)` | `v = momentum*v + lr*g; param -= v` (or Nesterov look-ahead) | "Heavy ball" style; `decay` gives `lr_eff = lr/(1+decay*iterations)`; `nesterov=True` requires `momentum > 0` |
| `Momentum(lr, beta=0.9)` | `v = beta*v + (1-beta)*g; param -= lr*v` | EMA-style velocity — a genuinely different rule from `SGD(momentum=b)` despite the shared hyperparameter name (see Design Notes) |
| `AdaGrad(lr, epsilon=1e-8)` | `cache += g**2; param -= lr*g/(sqrt(cache)+eps)` | `cache` only accumulates — effective LR shrinks monotonically |
| `RMSProp(lr=0.001, beta=0.9, epsilon=1e-8)` | `cache = beta*cache + (1-beta)*g**2; param -= lr*g/(sqrt(cache)+eps)` | Exponentially-decaying accumulator fixes AdaGrad's monotonic-shrink issue |
| `Adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)` | Bias-corrected first/second moment estimates; `param -= lr*m_hat/(sqrt(v_hat)+eps)` | Textbook/PyTorch-compatible form (epsilon applied to `sqrt(v_hat)`, not the paper's alternate combined-alpha reformulation) |

`AdaGrad`/`RMSProp`/`Adam` all route element-wise operations (squaring, division) through `_elementwise`/`_elementwise2`, which dispatch to `Vector.element_wise`/`Matrix.element_wise` for tensor-valued gradients — critical for `Matrix` params, where a naive `x * x` would invoke `Matrix.__mul__` (matrix multiplication) instead of squaring each entry. `_safe_sqrt` clamps to `sqrt(max(x, 0))` to absorb floating-point noise in accumulators that are mathematically guaranteed non-negative.

Each optimizer exposes `reset()` (clears its own accumulator state in addition to the base class's `iterations`/`history`) and `get_config()` (includes its own hyperparameters alongside `lr`).

### `second_order.py`

Both classes below operate on a **flat list of floats** and **return a new list** from `step()` (they don't mutate `params` in place). They use lightweight list-of-lists matrix helpers rather than `Matrix`, deliberately avoiding `Matrix`'s per-operation validation overhead in BFGS's hot inner loop.

```python
class NewtonMethod(Optimizer):
    def __init__(self, f, lr=1.0, hessian_h=1e-4)
    def step(self, params, grads) -> List[float]                    # numerical Hessian via utils.numerical_hessian
    def step_with_hessian(self, params, grads, hessian) -> List[float]  # caller-supplied Hessian
```
Damped Newton's method: `x -= lr * H⁻¹ @ grad`, solved via `_solve_linear` (Gauss-Jordan on the augmented system, not an explicit inverse). Checks the Newton direction's dot product with the gradient; if the Hessian has negative curvature and the raw direction would be an *ascent* direction, falls back to plain gradient descent with a logged warning rather than silently taking a bad step. Also falls back (with a warning) if the Hessian is singular.

```python
class BFGS(Optimizer):
    def __init__(self, lr=1.0, eps=1e-10, max_step=10.0)
    def step(self, params, grads) -> List[float]
```
Quasi-Newton method building an inverse-Hessian approximation from gradient history alone (Nocedal & Wright eq. 6.17). The update is applied only when the curvature condition `s·y > eps` holds **strictly positive** (not `abs(s·y) > eps`) — required to keep the inverse-Hessian approximation positive definite; allowing negative curvature through can turn the BFGS descent direction into an ascent direction. Steps are clamped to `max_step` Euclidean norm to guard against divergence while the inverse-Hessian approximation is still poor early in optimization.

Module-level linear algebra helpers (all list-of-lists based): `_solve_linear(A, b, tol=1e-12)` (Gauss-Jordan with a magnitude-scaled pivot tolerance), `_validate_square_system`, `_identity(n)`, `_mat_mat`, `_outer`, `_mat_scale`, `_mat_add`, `_mat_sub`.

### `line_search.py`

```python
backtracking(f, x, grad, direction, alpha=1.0, rho=0.5, c=1e-4, max_iter=100) -> float
```
Armijo backtracking: shrinks `alpha` by `rho` until `f(x + alpha*direction) <= f(x) + c*alpha*(grad·direction)`. Raises `ValueError` if `direction` isn't a descent direction (`grad·direction >= 0`), or if `alpha`/`rho`/`c`/`max_iter` are out of range. Returns the smallest step tried (with a logged warning) if `max_iter` is exhausted without satisfying the condition. Objective-function exceptions are wrapped in `RuntimeError` with context.

```python
golden_section(f, a, b, tol=1e-8, max_iter=200) -> dict
```
Golden-section search for the minimum of a unimodal 1D function on `[a, b]`. Returns `{"x_min", "f_min", "bracket", "converged"}`. Requires `a < b`, both finite; raises `ValueError` otherwise, or for non-positive `tol`/`max_iter`.

### `utils.py`

```python
numerical_gradient(f, x, h=1e-5) -> List[float]
```
Central-difference gradient, `2*len(x)` evaluations of `f`. Defensively copies `x` to a plain list first — `x[:]` is a *view*, not a copy, for NumPy arrays, so without this, in-place perturbation during finite differencing would silently corrupt the caller's original array (relevant when interoperating with `scipy.optimize`, which passes NumPy arrays to `jac`/`hess`).

```python
numerical_hessian(f, x, h=1e-4) -> List[List[float]]
```
Central-difference Hessian: 3-point diagonal formula, 4-point mixed-partial off-diagonal formula, exploiting symmetry to evaluate each off-diagonal pair once — `2n² + 1` evaluations instead of a naive `4n²` (which would also inconsistently use step `2h` instead of `h` on the diagonal). Logs a warning above `n=50` parameters (O(n²) cost). Same NumPy-array defensive-copy behavior as `numerical_gradient`.

```python
class ConvergenceTracker:
    def __init__(self, tol=1e-6, patience=10)
    def update(self, loss: float) -> bool          # True if no improvement for `patience` consecutive calls
    def converged(self) -> bool                     # last two losses differ by < tol
    def reset(self) -> None
    def plot_ascii(self, width=60, height=12) -> None
```
Plateau-based early-stopping signal. `converged()` is a simple two-point check (documented as weaker than `update`'s patience-based signal — can false-positive on a loss oscillating within a band smaller than `tol`). `plot_ascii` prints a crude ASCII bar chart of loss history; raises `ValueError` for `height < 2` (avoids a division-by-zero when computing row thresholds).

```python
optimize(f, grad_f, x0, optimizer, max_iter=1000, tol=1e-6, patience=20, verbose=False) -> dict
```
Generic driver loop: runs `optimizer` on `f` from `x0` until the gradient norm drops below `tol` or the loss plateaus for `patience` iterations (via `ConvergenceTracker`), or `max_iter` is reached. Correctly dispatches both optimizer calling conventions: `result = optimizer.step(x, grads); if result is not None: x = result`. Returns `{"x", "loss", "iterations", "history", "converged", "stop_reason"}`. Raises `ValueError` for empty `x0` or invalid `max_iter`/`tol`, `TypeError` if `optimizer` lacks a callable `step`.

Also exposes flat-vector helpers used internally and by `second_order.py`: `_vec_add`, `_vec_sub`, `_vec_scale`, `_vec_dot`, `_vec_norm`, `_mat_vec` — deliberately plain-list based (no `Vector`/`Matrix` validation overhead) for use in hot loops.

---

## Example Session

```python
from first_order import GradientDescent, Adam, RMSProp
from second_order import NewtonMethod, BFGS
from line_search import backtracking, golden_section
from utils import numerical_gradient, optimize, ConvergenceTracker

# First-order: minimize f(x) = x^2
opt = Adam(lr=0.1)
params = [10.0]
for _ in range(200):
    grad = [2 * params[0]]
    opt.step(params, grad)
round(params[0], 2)                     # 0.0

# Generic driver loop (works with either optimizer family)
f = lambda x: sum(xi**2 for xi in x)
grad_f = lambda x: numerical_gradient(f, x)
result = optimize(f, grad_f, [5.0, -3.0], GradientDescent(lr=0.1), max_iter=500)
result["loss"], result["converged"], result["stop_reason"]

result_bfgs = optimize(f, grad_f, [5.0, -3.0], BFGS(lr=1.0), max_iter=100)

# Second-order: Newton's method on a quadratic bowl
g = lambda x: (x[0] - 3)**2 + 2*(x[1] + 1)**2
nm = NewtonMethod(g, lr=1.0)
x = [0.0, 0.0]
for _ in range(5):
    x = nm.step(x, numerical_gradient(g, x))
round(x[0], 2), round(x[1], 2)          # (3.0, -1.0)

# Line search
alpha = backtracking(lambda x: x[0]**2, [3.0], grad=[6.0], direction=[-1.0])
gs = golden_section(lambda x: (x - 2.0)**2 + 1.0, 0.0, 5.0)
```

---

## Design Notes

- **Two calling conventions, one driver loop.** First-order optimizers
  mutate `params` in place (matching the common deep-learning idiom of
  updating parameter tensors directly); second-order optimizers return a
  new list because Newton's method/BFGS need the *entire* current vector
  at once to solve a linear system or apply an inverse-Hessian
  approximation — an in-place per-element update doesn't make sense for
  them. `utils.optimize()` handles both transparently by checking
  whether `step()`'s return value is `None`. This contract is documented
  once, in `base.py`'s module docstring, and referenced everywhere else
  rather than re-explained.
- **`SGD(momentum=b)` and `Momentum(beta=b)` are deliberately different
  update rules** despite the shared hyperparameter name/value. `SGD`'s
  momentum ("heavy ball") accumulates raw `lr*g` terms in its velocity;
  `Momentum`'s velocity is an exponential moving average of the raw
  gradient `g` (structurally closer to Adam's first moment, without bias
  correction). Both are standard in different textbooks/frameworks; a
  regression test (`test_differs_from_sgd_momentum_for_same_beta`)
  explicitly guards against them accidentally converging to the same
  trajectory.
- **Element-wise operations must route through `Vector`/`Matrix`'s own
  `element_wise` methods for tensor-valued gradients.** `AdaGrad`,
  `RMSProp`, and `Adam` all need to square gradients and divide
  element-wise; naively calling `grad * grad` on a `Matrix` would invoke
  `Matrix.__mul__` (real matrix multiplication) instead of squaring each
  entry — a completely different (and shape-incompatible, for
  non-square gradients) operation. `_elementwise`/`_elementwise2`
  dispatch correctly based on type, and a regression test
  (`test_matrix_gradient_elementwise_square_not_matmul`) locks this in.
- **Adam uses the textbook/PyTorch-compatible bias-correction formula**
  (explicit `m_hat`, `v_hat`, with `epsilon` applied to `sqrt(v_hat)`)
  rather than the paper's alternate single-combined-learning-rate
  reformulation. The two are only "almost equivalent" — they diverge
  when `epsilon` is non-negligible relative to the gradient — and this
  implementation deliberately matches what "Adam" means in practice for
  anyone comparing against PyTorch/TensorFlow. Guarded by
  `test_matches_textbook_formula_manually` with a large `epsilon` chosen
  specifically to make the two formulas visibly diverge if the wrong one
  were used.
- **Newton's method detects and recovers from non-descent directions.**
  A raw Newton step (`H⁻¹ @ grad`) assumes a positive-definite Hessian;
  near a saddle point or in a negative-curvature region, the "Newton
  direction" can actually point *uphill*. Rather than silently taking
  a step that increases the loss, `_newton_direction` checks
  `direction · grad > 0` and falls back to plain gradient descent (with
  a logged warning) if it fails — a lightweight safeguard well short of
  a full modified-Newton method (eigenvalue clamping,
  Levenberg-Marquardt damping), which is explicitly out of scope.
- **BFGS's curvature condition is strict (`s·y > eps`), not
  `abs(s·y) > eps`.** Allowing strongly *negative* curvature through the
  update would corrupt the positive-definiteness of the inverse-Hessian
  approximation, which can flip the BFGS search direction into an
  ascent direction. This was a real bug class, not a hypothetical —
  `test_curvature_condition_preserves_positive_definiteness` constructs
  a specific negative-curvature scenario and verifies `H_inv` is left
  untouched (still positive definite) rather than corrupted.
- **`_solve_linear` solves the augmented system directly rather than
  computing `A⁻¹` and multiplying** — roughly half the arithmetic (`n+1`
  augmented columns vs. `2n` for a full inversion), which matters since
  Newton's method calls this every iteration. Its pivot tolerance is
  scaled by the matrix's own magnitude, the same pattern used in
  `Matrix.eigenvectors` for large-magnitude systems (e.g.
  high-curvature Hessians).
- **`numerical_hessian` exploits symmetry and gets the diagonal step
  size right.** A naive implementation might apply the 4-point
  mixed-partial formula uniformly to every `(i, j)` including the
  diagonal, which silently uses an effective step of `2h` instead of
  `h` there (because `+=`/`-=` compose on the same index) and wastes
  evaluations re-computing `f(x)` redundantly. The dedicated diagonal
  formula and off-diagonal symmetry exploitation together roughly halve
  the evaluation count (`2n²+1` vs. a naive `4n²`) — both effects are
  covered by dedicated regression tests.
- **NumPy-array safety in finite differences.** `numerical_gradient`
  and `numerical_hessian` both defensively copy their input to a plain
  Python list before perturbing coordinates, since `x[:]` is a *view*
  (not a copy) for NumPy arrays — without this, in-place perturbation
  during finite-differencing would silently corrupt the caller's
  original array. This matters directly for interoperating with
  `scipy.optimize`, which passes NumPy arrays to `jac`/`hess` callbacks;
  both functions are tested for exact compatibility with
  `scipy.optimize.minimize(method="Newton-CG")`.
- **`ConvergenceTracker.plot_ascii` guards its own division.** `height`
  is used as a divisor (`row / (height - 1)`) when computing per-row
  thresholds; `height < 2` is rejected up front with a clear
  `ValueError` rather than raising an opaque `ZeroDivisionError`.
- **`optimize()`'s early-stopping is genuinely wired up**, not just
  computed and discarded — `ConvergenceTracker.update()`'s return value
  is checked every iteration and triggers a `"plateau"` stop reason,
  verified by a dedicated regression test using a constant loss and a
  gradient that never drops below the gradient-norm threshold (isolating
  the plateau-detection path specifically).

---

## Test Coverage

Each library file has a matching `tests/test_*.py`. `test_line_search.py`
and `test_second_order.py` cross-check against `scipy.optimize`
(`minimize_scalar`, `minimize(method="Newton-CG"/"BFGS")`, including on
the Rosenbrock function). `test_first_order.py` verifies convergence on
`f(x)=x²` for every optimizer plus optimizer-specific behavioral
properties (Adam's first-step-≈-lr property, AdaGrad's monotonically
shrinking step size, the Matrix-gradient elementwise-not-matmul fix).
`test_utils.py` and `test_second_order.py` include NumPy-array
view-vs-copy regression tests. `test_base.py` covers the shared
`Optimizer` contract and `_zeros_like` across `float`/`Vector`/`Matrix`.

Run the full suite with verbose output:

```bash
pytest tests/ -v
```

---

## Roadmap Context

`Optimization/` is the fifth module in **Phase 1: Math Foundations**,
building on `Vectors/vector.py` and `Matrix/matrix.py` for
tensor-valued first-order optimizer parameters. It completes the
numerical-methods foundation — linear algebra, probability/statistics,
Bayesian inference, and now optimization — that later phases (from-scratch
neural networks, classical ML algorithms trained via gradient descent)
will depend on directly.
