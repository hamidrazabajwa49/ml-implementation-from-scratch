# ML_Models / svm — Pure Python Support Vector Machines

## Overview

The `ML_Models/svm/` sub-package implements soft-margin Support Vector Machines from scratch — part of the *Engineering Redemption Arc*, a structured 60-project ML engineering curriculum in the [`ml-implementation-from-scratch`](https://github.com/) repository.

Two model classes are provided: `LinearSVM`, a primal formulation trained by subgradient descent, and `KernelSVM`, a dual formulation solved via Simplified Sequential Minimal Optimization (SMO). Three kernels are available — linear, RBF, and polynomial — along with a resolver that accepts custom callables. Both models accept `Matrix` feature inputs and `Vector` targets in ±1 label format and share the `MLModels` base class interface. No NumPy, SciPy, CVXOPT, or scikit-learn is used.

---

## Project Structure

```
ml-implementation-from-scratch/
├── Vectors/
│   └── vector.py
├── Matrix/
│   └── matrix.py
├── Optimization/
│   └── first_order.py             # Adam (default optimizer for LinearSVM)
├── ML_Models/
│   ├── base_class.py              # MLModels abstract base class
│   ├── metrics.py                 # accuracy
│   └── svm/
│       ├── kernels.py             # linear_kernel, rbf_kernel, polynomial_kernel, _resolve_kernel
│       └── svm.py                 # LinearSVM, KernelSVM
```

---

## Dependencies

| Requirement | Detail |
|---|---|
| Python | 3.10+ |
| `math`, `os`, `sys`, `random`, `warnings`, `typing` | Standard library only |
| `Vectors/vector.py` | Feature rows, weight vectors, predictions |
| `Matrix/matrix.py` | Feature matrix input to `fit()` and `predict()` |
| `Optimization/first_order.py` | `Adam` — default optimizer for `LinearSVM` |
| `ML_Models/base_class.py` | `MLModels` base class |
| `ML_Models/metrics.py` | `accuracy` |
| `ML_Models/svm/kernels.py` | Kernel functions and resolver |

No `pip install` required.

---

## Installation

```python
import sys
sys.path.insert(0, "/path/to/ml-implementation-from-scratch")

from ML_Models.svm.kernels import linear_kernel, rbf_kernel, polynomial_kernel
from ML_Models.svm.svm import LinearSVM, KernelSVM
```

---

## Label Format

Both models require labels in **±1 format** — `+1.0` and `-1.0`. Standard 0/1 binary labels must be converted before calling `fit()`:

```python
y_pm1 = Vector([1.0 if v == 1 else -1.0 for v in y])
```

Passing any label outside `{-1, -1.0, 1, 1.0}` raises `ValueError`.

---

## Module Reference

---

### `kernels.py` — Kernel Functions

Three kernel functions and a resolver. All kernel functions accept two equal-length `Vector` arguments and return a `float`.

---

#### `linear_kernel(a, b)`

Dot product: `K(a, b) = aᵀb`.

```python
from ML_Models.svm.kernels import linear_kernel

linear_kernel(Vector([1.0, 2.0]), Vector([3.0, 4.0]))   # 11.0
```

---

#### `rbf_kernel(a, b, gamma=None)`

Radial basis function: `K(a, b) = exp(−γ ‖a − b‖²)`.

```python
from ML_Models.svm.kernels import rbf_kernel

rbf_kernel(Vector([1.0, 0.0]), Vector([0.0, 1.0]))            # gamma defaults to 1/d
rbf_kernel(Vector([1.0, 0.0]), Vector([0.0, 1.0]), gamma=0.5)
```

When `gamma=None`, it defaults to `1.0 / len(a)` (the standard `scale` heuristic). `gamma` must be strictly positive.

---

#### `polynomial_kernel(a, b, degree=3, coef0=1.0)`

Polynomial kernel: `K(a, b) = (aᵀb + coef0)^degree`.

```python
from ML_Models.svm.kernels import polynomial_kernel

polynomial_kernel(Vector([1.0, 2.0]), Vector([3.0, 4.0]), degree=2, coef0=1.0)   # 144.0
```

`degree` must be a positive `int`. `coef0` controls the influence of lower-degree terms.

---

#### `_resolve_kernel(kernel, gamma=None, degree=3, coef0=1.0)`

Resolves a kernel spec into a `(Vector, Vector) -> float` callable. Used internally by `KernelSVM`.

```python
from ML_Models.svm.kernels import _resolve_kernel

fn = _resolve_kernel('linear')
fn = _resolve_kernel('rbf', gamma=0.5)
fn = _resolve_kernel('poly', degree=2, coef0=0.0)
fn = _resolve_kernel(my_kernel_fn)          # callable passed through unchanged
```

| Input | Behavior |
|---|---|
| `'linear'` | Returns a closure over `linear_kernel` |
| `'rbf'` | Returns a closure over `rbf_kernel` with the given `gamma` |
| `'poly'` | Returns a closure over `polynomial_kernel` with the given `degree` and `coef0` |
| Any other string | Raises `ValueError` with list of valid options |
| Callable | Returned as-is |
| Anything else | Raises `TypeError` |

`gamma`, `degree`, and `coef0` are captured at resolution time. Changing them after `KernelSVM.__init__` has no effect.

---

### `svm.py` — SVM Models

---

### `LinearSVM`

Soft-margin linear SVM trained in the primal via subgradient descent on the hinge loss with L2 regularization. The objective minimized is:

```
(1/2) ‖w‖² + C · (1/m) · Σ max(0, 1 − yᵢ (wᵀxᵢ + b))
```

Larger `C` increases the penalty for margin violations and drives the model toward hard-margin behavior.

```python
from ML_Models.svm.svm import LinearSVM

model = LinearSVM(C=1.0, lr=0.01, n_iter=1000)
model.fit(X_train, y_train)

scores = model.decision_function(X_test)    # Vector of raw margin scores
labels = model.predict(X_test)              # Vector of +1.0 / -1.0
acc    = model.score(X_test, y_test)        # accuracy (float)
w      = model.parameters()                # [Vector] — weights including bias at index 0
```

| Parameter | Default | Constraint |
|---|---|---|
| `C` | `1.0` | Positive number |
| `lr` | `0.01` | Positive number |
| `n_iter` | `1000` | Positive `int` |

**`fit(X, y, optimizer=None)`**

Trains the model. Bias is handled by prepending a column of 1s to `X` via `_add_bias_column`; `self.w[0]` is the bias and is excluded from the L2 regularization penalty.

A custom optimizer from `Optimization/` can be passed. If `None`, `Adam(lr=self.lr)` is used. If a pre-warmed optimizer instance is passed (one that has been used in a previous `fit()` call), its internal state is reset and a `UserWarning` is issued.

Loss (regularization + hinge) is recorded every 100 iterations and on the final iteration, stored in both `optimizer.history` and `self.loss_history`.

```python
from Optimization.first_order import SGD

model.fit(X_train, y_train, optimizer=SGD(lr=0.005, momentum=0.9))
print(model.loss_history)
```

**`decision_function(X)`**

Returns the raw signed margin `wᵀx + b` for each row of `X` as a `Vector`. Positive scores predict `+1`, negative scores predict `−1`.

**`predict(X)`**

Returns a `Vector` of `+1.0` / `-1.0` labels based on the sign of `decision_function(X)`. Scores exactly `0.0` are assigned `+1.0`.

**`score(X, y)`**

Returns accuracy as a `float` in `[0.0, 1.0]`.

**`parameters()`**

Returns `[self.w]` — a single-element list containing the weight `Vector` of length `n_features + 1`. Raises `RuntimeError` if called before `fit()`.

---

### `KernelSVM`

Soft-margin kernel SVM solved in the dual via Simplified SMO. The dual maximizes:

```
W(α) = Σ αᵢ − (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
subject to: 0 ≤ αᵢ ≤ C, Σ αᵢ yᵢ = 0
```

Prediction at inference time uses only the support vectors (training points with `αᵢ > 1e-7`), making inference fast even for large training sets. The full Gram matrix is precomputed during `fit()` and stored for `dual_objective()`.

```python
from ML_Models.svm.svm import KernelSVM

model = KernelSVM(C=1.0, kernel='rbf', gamma=0.5)
model.fit(X_train, y_train)

scores = model.decision_function(X_test)    # Vector of raw scores
labels = model.predict(X_test)              # Vector of +1.0 / -1.0
acc    = model.score(X_test, y_test)        # accuracy (float)
obj    = model.dual_objective()             # float — SMO dual objective value
params = model.parameters()                 # dict — alphas, b, n_support
```

| Parameter | Default | Constraint |
|---|---|---|
| `C` | `1.0` | Positive number |
| `kernel` | `'rbf'` | `'linear'`, `'rbf'`, `'poly'`, or callable |
| `gamma` | `None` | Positive float; `None` → `1/d` (RBF only) |
| `degree` | `3` | Positive `int` (poly only) |
| `coef0` | `1.0` | Any float (poly only) |
| `tol` | `1e-3` | Positive number — KKT violation tolerance |
| `max_passes` | `10` | Positive `int` — consecutive passes without an alpha update before stopping |
| `random_state` | `None` | `int` or `None` — seed for random pair selection in SMO |

**`fit(X, y)`**

Trains the model via Simplified SMO. Requires at least 2 training samples and both classes present in `y`.

The Gram matrix `K[i][j] = kernel(xᵢ, xⱼ)` is precomputed in O(m²) kernel evaluations and cached. SMO then iterates: for each sample `i` that violates the KKT conditions (within tolerance `tol`), a random `j ≠ i` is chosen, the pair `(αᵢ, αⱼ)` is optimized in closed form, and the bias `b` is updated. A pass with no alpha changes increments the `passes` counter; the loop terminates when `passes == max_passes`.

After fitting, only support vectors (`αᵢ > 1e-7`) are retained in `self.support_X`, `self.support_y`, and `self.alphas`. If zero support vectors remain, a `UserWarning` is issued.

**`decision_function(X)`**

Computes `f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b` over support vectors only. Returns a `Vector`.

**`predict(X)`**

Returns `+1.0` where `decision_function >= 0`, `-1.0` otherwise.

**`score(X, y)`**

Returns accuracy as a `float`.

**`dual_objective()`**

Returns `W(α) = Σ αᵢ − (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)` over all training points (not just support vectors). Higher is better. Useful for diagnosing whether SMO made meaningful progress.

```python
print(model.dual_objective())
```

**`parameters()`**

Returns a `dict`:

```python
{
    "alphas":    [float, ...],   # one per support vector
    "b":         float,          # bias term
    "n_support": int             # number of support vectors
}
```

Raises `RuntimeError` if called before `fit()`.

---

## Example Session

```python
from Vectors.vector import Vector
from Matrix.matrix import Matrix
from ML_Models.svm.svm import LinearSVM, KernelSVM
from ML_Models.svm.kernels import rbf_kernel, polynomial_kernel
from Optimization.first_order import SGD

# ±1 label data
X = Matrix([
    [1.0, 2.0], [1.5, 1.8], [2.0, 2.5],
    [5.0, 6.0], [5.5, 5.5], [6.0, 6.5]
])
y = Vector([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

# --- LinearSVM ---
lin = LinearSVM(C=1.0, lr=0.01, n_iter=1000)
lin.fit(X, y)

print(lin.predict(X))                    # Vector of ±1.0
print(lin.score(X, y))                   # accuracy
print(lin.decision_function(X))          # raw margin scores
print(lin.loss_history)                  # hinge + L2 loss at recorded steps
print(lin.parameters())                  # [Vector of weights]

# Custom optimizer
lin2 = LinearSVM(C=0.5, lr=0.005, n_iter=2000)
lin2.fit(X, y, optimizer=SGD(lr=0.005, momentum=0.9))
print(lin2.score(X, y))

# --- KernelSVM (RBF) ---
rbf = KernelSVM(C=1.0, kernel='rbf', gamma=0.5, tol=1e-3, max_passes=10)
rbf.fit(X, y)

print(rbf.predict(X))
print(rbf.score(X, y))
print(rbf.dual_objective())
print(rbf.parameters())                  # {'alphas': [...], 'b': float, 'n_support': int}

# --- KernelSVM (Polynomial) ---
poly = KernelSVM(C=1.0, kernel='poly', degree=2, coef0=1.0, max_passes=15)
poly.fit(X, y)
print(poly.score(X, y))

# --- KernelSVM (Linear kernel) ---
ksvm_lin = KernelSVM(C=1.0, kernel='linear')
ksvm_lin.fit(X, y)
print(ksvm_lin.score(X, y))

# --- KernelSVM (Custom kernel) ---
ksvm_custom = KernelSVM(C=1.0, kernel=lambda a, b: rbf_kernel(a, b, gamma=0.1))
ksvm_custom.fit(X, y)
print(ksvm_custom.predict(X))

# --- Convert 0/1 labels to ±1 ---
y_binary = Vector([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
y_pm1    = Vector([1.0 if v == 1.0 else -1.0 for v in y_binary])
lin.fit(X, y_pm1)
```

---

## Error Reference

| Situation | Exception |
|---|---|
| `y` contains labels outside `{-1, -1.0, 1, 1.0}` | `ValueError` |
| `C`, `lr`, or `tol` not a positive number | `ValueError` / `TypeError` |
| `C`, `lr`, or `tol` is a `bool` | `TypeError` |
| `n_iter` or `max_passes` not a positive `int` | `ValueError` / `TypeError` |
| `gamma <= 0` in RBF kernel | `ValueError` |
| `degree < 1` or `degree` not an `int` in polynomial kernel | `ValueError` / `TypeError` |
| Unknown kernel string | `ValueError` |
| `kernel` not a string or callable | `TypeError` |
| `random_state` not an `int` or `None` | `TypeError` |
| `X.n_rows != len(y)` | `ValueError` |
| Empty `X` or `y` | `ValueError` |
| Fewer than 2 training samples in `KernelSVM` | `ValueError` |
| Only one class in `y` for `KernelSVM` | `ValueError` |
| `predict()`, `score()`, or `parameters()` before `fit()` | `RuntimeError` |
| `X` not a `Matrix` in `decision_function()` | `TypeError` |
| Vectors of unequal length in any kernel function | `ValueError` |
| Empty vectors in `rbf_kernel` | `ValueError` |

---

## Design Notes

- **`LinearSVM` uses the primal; `KernelSVM` uses the dual:** The primal formulation gives direct access to the weight vector `w`, making `decision_function` an O(d) dot product. The dual formulation supports arbitrary kernels but inference scales with the number of support vectors.
- **Bias excluded from L2 regularization in `LinearSVM`:** `_add_bias_column` prepends a column of 1s; `self.w[0]` is the bias. The regularization gradient loop starts at index 1, so the bias is not penalized.
- **Optimizer state reset warning in `LinearSVM`:** Reusing a warmed optimizer across multiple `fit()` calls would carry over stale moment estimates. The check on `optimizer.t` detects this and resets `t`, `m`, `v` before training begins, issuing a `UserWarning` so the caller is aware.
- **Gram matrix precomputed in `KernelSVM`:** All `m²` kernel evaluations are computed once at the start of `fit()` and stored in `self._K`. SMO then accesses `K[i][j]` in O(1) per lookup. This trades memory (O(m²) floats) for speed during the optimization passes.
- **Only support vectors retained after `KernelSVM.fit()`:** Post-training, `self.support_X`, `self.support_y`, and `self.alphas` contain only the `nₛᵥ` entries where `αᵢ > 1e-7`. The full `_all_alpha`, `_all_ys`, and `_K` are kept separately for `dual_objective()` only.
- **SMO pair selection is random:** The inner loop picks `j` uniformly at random from `{0, …, m-1} \ {i}`. Seeding via `random_state` makes results reproducible. More sophisticated heuristics (e.g., maximum violator selection) would reduce passes but add complexity.
- **`dual_objective()` operates on all training alphas:** Unlike inference, which uses only support vectors, `dual_objective()` sums over all `m` training points using the cached `_K` and `_all_alpha`. Skipping zero alphas early avoids the full O(m²) loop in sparse solutions.
- **Zero support vectors warning:** When `C` is very small or the kernel is poorly scaled, SMO may leave every `αᵢ ≈ 0`. The model degenerates to predicting based solely on the bias `b`. A `UserWarning` is issued rather than raising an exception, since the fitted object is technically valid.

---

## Roadmap Context

This sub-package depends on:

- **`Vectors/`** — Project 1
- **`Matrix/`** — Project 2
- **`Optimization/`** — Project 10 (`Adam` and the optimizer abstraction, used by `LinearSVM`)
- **`ML_Models/base_class.py`** and **`ML_Models/metrics.py`** — established by the `linear/` sub-package (Project 11)

`LinearSVM` demonstrates that the gradient-based optimizer abstraction from `Optimization/` extends naturally to non-smooth losses (hinge loss via subgradients). `KernelSVM` introduces the dual formulation and the kernel trick, which are foundational concepts for the kernel methods and non-linear models in subsequent projects.
