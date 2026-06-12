# ML_Models / Linear — Pure Python Linear & Logistic Regression

## Overview

The `ML_Models/linear/` sub-package implements linear and logistic regression models from scratch — the first machine learning algorithms in the *Engineering Redemption Arc*, a structured 60-project ML engineering curriculum in the [`ml-implementation-from-scratch`](https://github.com/) repository.

Four regression variants and three classifier variants are provided, sharing a common base class and evaluation infrastructure defined in `ML_Models/`. All models accept `Matrix` feature inputs and `Vector` targets, integrate directly with the `Optimization/` package for gradient-based training, and use `InformationTheory/` for loss computation. No NumPy, SciPy, or scikit-learn is used.

---

## Project Structure

```
ml-implementation-from-scratch/
├── Vectors/
│   └── vector.py
├── Matrix/
│   └── matrix.py
├── Optimization/
│   └── first_order.py             # Adam (default optimizer)
├── InformationTheory/
│   └── information_theory.py      # binary_cross_entropy
└── ML_Models/
    ├── base_class.py              # MLModels abstract base class
    ├── metrics.py                 # Regression and classification metrics
    ├── activations.py             # sigmoid (element-wise, polymorphic)
    └── linear/
        ├── linear_regression.py   # LinearRegression, GDLinearRegression,
        │                          # RidgeRegression, LassoRegression
        └── logistic_regression.py # LogisticRegression, LogisticRegressionL2,
                                   # LogisticRegressionL1
```

---

## Dependencies

| Requirement | Detail |
|---|---|
| Python | 3.10+ |
| `math`, `os`, `sys`, `typing` | Standard library only |
| `Vectors/vector.py` | Feature rows, weight vectors, predictions |
| `Matrix/matrix.py` | Feature matrix, design matrix with bias column |
| `Optimization/first_order.py` | `Adam` — default optimizer for gradient-based models |
| `InformationTheory/information_theory.py` | `binary_cross_entropy` — logistic training loss |
| `ML_Models/base_class.py` | `MLModels` base class |
| `ML_Models/metrics.py` | `r2_score`, `accuracy` |
| `ML_Models/activations.py` | `sigmoid` |

No `pip install` required.

---

## Installation

```python
import sys
sys.path.insert(0, "/path/to/ml-implementation-from-scratch")

from ML_Models.linear.linear_regression import (
    LinearRegression, GDLinearRegression, RidgeRegression, LassoRegression
)
from ML_Models.linear.logistic_regression import (
    LogisticRegression, LogisticRegressionL2, LogisticRegressionL1
)
```

---

## Shared Infrastructure

### `base_class.py` — `MLModels`

Abstract base class for all models. Defines the standard interface and shared helpers.

```python
model.fit(X, y)         # train the model
model.predict(X)        # return predictions as a Vector
model.score(X, y)       # return a scalar metric (R² or accuracy)
model.parameters()      # return [self.w] once fitted
```

**Helpers (called internally by all subclasses):**

| Method | Purpose |
|---|---|
| `_validate_Xy(X, y)` | Checks `X` is `Matrix`, `y` is `Vector`, shapes match, both non-empty |
| `_check_is_fitted()` | Raises `RuntimeError` if `self.w` does not exist |
| `_add_bias_column(X)` | Prepends a column of 1s to `X`, returning an augmented `Matrix` |

---

### `metrics.py` — Evaluation Metrics

All metric functions accept `Vector` arguments. `y_true` and `y_pred` must have the same non-zero length.

#### Regression metrics

```python
from ML_Models.metrics import mse, mae, rmse, r2_score

mse(y_true, y_pred)      # mean squared error
mae(y_true, y_pred)      # mean absolute error
rmse(y_true, y_pred)     # root mean squared error
r2_score(y_true, y_pred) # coefficient of determination
```

`r2_score` returns `0.0` when both SS_res and SS_tot are zero (constant predictions on a constant target), and `float('-inf')` when SS_tot is zero but SS_res is not.

#### Classification metrics

```python
from ML_Models.metrics import (
    accuracy, accuracy_score, precision, recall,
    f1_score, confusion_matrix, roc_auc
)

accuracy(y_true, y_pred)         # fraction of correct predictions
precision(y_true, y_pred)        # TP / (TP + FP)
recall(y_true, y_pred)           # TP / (TP + FN)
f1_score(y_true, y_pred)         # harmonic mean of precision and recall
confusion_matrix(y_true, y_pred) # {"TP": int, "TN": int, "FP": int, "FN": int}
roc_auc(y_true, y_proba)         # trapezoidal AUC; handles tied scores correctly
```

`accuracy` and `accuracy_score` are identical functions — both are provided. `precision`, `recall`, `f1_score`, `confusion_matrix`, and `roc_auc` require binary `y_pred` values (`0` or `1`); non-binary values raise `ValueError`.

`roc_auc` batches tied probability scores before accumulating the trapezoidal area, preventing AUC inflation from ties.

---

### `activations.py` — `sigmoid`

Numerically stable sigmoid, dispatched element-wise over `float`, `Vector`, or `Matrix`.

```python
from ML_Models.activations import sigmoid

sigmoid(0.0)                    # 0.5
sigmoid(Vector([0.0, 2.0]))     # Vector of probabilities
sigmoid(Matrix([[0.0, 1.0]]))   # Matrix of probabilities
```

Uses the `exp(z) / (1 + exp(z))` form for negative inputs to avoid overflow.

---

## Linear Regression Models

All four models share the `fit → predict → score → parameters` interface. Weights are stored in `self.w` as a `Vector` of length `n_features + 1` (bias at index 0).

---

### `LinearRegression`

Closed-form Ordinary Least Squares: `w = (AᵀA)⁻¹ Aᵀy`.

```python
from ML_Models.linear.linear_regression import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)        # Vector
r2     = model.score(X_test, y_test)  # float
w      = model.parameters()           # [Vector]
```

`_regularized_inverse` adds `tol × I` before inverting if `|det(AᵀA)| < tol` (default `1e-10`), preventing failure on nearly singular design matrices.

---

### `GDLinearRegression`

Gradient descent with MSE loss. Defaults to Adam; accepts any optimizer from `Optimization/`.

```python
from ML_Models.linear.linear_regression import GDLinearRegression
from Optimization.first_order import SGD

model = GDLinearRegression()
model.fit(X_train, y_train, n_iter=2000, lr=0.01)

# With a custom optimizer:
model.fit(X_train, y_train, optimizer=SGD(lr=0.005, momentum=0.9), n_iter=3000)
```

Gradient: `g = (2/m) · Aᵀ(Aw − y)`. Loss is recorded to `optimizer.history` every 100 iterations and on the final iteration.

| Parameter | Default | Constraint |
|---|---|---|
| `optimizer` | `Adam(lr=lr)` | Any `Optimizer` subclass |
| `n_iter` | `1000` | `>= 1` |
| `lr` | `0.01` | `> 0` |

---

### `RidgeRegression`

L2-regularized closed-form: `w = (AᵀA + λI*)⁻¹ Aᵀy`, where `I*` is the identity with the top-left (bias) entry zeroed — bias is not penalized.

```python
from ML_Models.linear.linear_regression import RidgeRegression

model = RidgeRegression()
model.fit(X_train, y_train, lam=1.0)
```

`lam` must be non-negative. Setting `lam=0` reduces to `LinearRegression` (without the near-singularity guard).

---

### `LassoRegression`

L1-regularized fitting via coordinate descent with soft-thresholding.

```python
from ML_Models.linear.linear_regression import LassoRegression

model = LassoRegression()
model.fit(X_train, y_train, lam=0.1, n_iters=1000, tol=1e-4)
```

Each iteration cycles over features `j = 1, …, d` (bias `w[0]` updated separately without penalty). The soft-threshold operator shrinks each coordinate:

```
soft_threshold(z, α) = z − α  if z > α
                       z + α  if z < −α
                       0      otherwise
```

The effective threshold per step is `lam × m`. Convergence is declared when `max |Δw_j| < tol` across a full cycle.

| Parameter | Default | Constraint |
|---|---|---|
| `lam` | `0.1` | `>= 0` |
| `n_iters` | `1000` | `>= 1` |
| `tol` | `1e-4` | `> 0` |

---

## Logistic Regression Models

All three classifiers share the same `fit → predict_proba → predict → score → parameters` interface. Weights in `self.w` are a `Vector` of length `n_features + 1`.

---

### `LogisticRegression`

Binary classifier trained with binary cross-entropy loss and Adam optimizer (default).

```python
from ML_Models.linear.logistic_regression import LogisticRegression

model = LogisticRegression(lr=0.01, n_iter=1000)
model.fit(X_train, y_train)

probs  = model.predict_proba(X_test)         # Vector of probabilities in (0, 1)
labels = model.predict(X_test, threshold=0.5) # Vector of 0.0 / 1.0
acc    = model.score(X_test, y_test)          # accuracy
```

Gradient: `g = (1/m) · Aᵀ(σ(Aw) − y)`. Loss (`binary_cross_entropy`) recorded every 100 iterations and on the final iteration. A custom optimizer can be passed to `fit()`:

```python
from Optimization.first_order import SGD
model.fit(X_train, y_train, optimizer=SGD(lr=0.01, momentum=0.9))
```

`predict()` threshold defaults to `0.5`; must be in `(0, 1)`.

---

### `LogisticRegressionL2`

Inherits `LogisticRegression`. Adds L2 penalty `(λ/2m) Σ w_j²` (bias excluded) to the gradient and recorded loss.

```python
from ML_Models.linear.logistic_regression import LogisticRegressionL2

model = LogisticRegressionL2(lam=1.0, lr=0.01, n_iter=1000)
model.fit(X_train, y_train)
```

L2 gradient addition: `g_j += (λ/m) · w_j` for `j ≥ 1`. Bias weight `w_0` is not penalized. Uses Adam by default; accepts a custom optimizer.

`lam` must be non-negative.

---

### `LogisticRegressionL1`

Inherits `LogisticRegression`. Applies L1 regularization via a proximal gradient step (soft-thresholding) rather than adding to the gradient directly.

```python
from ML_Models.linear.logistic_regression import LogisticRegressionL1

model = LogisticRegressionL1(lam=0.1, lr=0.01, n_iter=1000)
model.fit(X_train, y_train)

print(model.loss_history)   # BCE + L1 penalty recorded every 100 iters
```

Per-step update: `w_j ← soft_threshold(w_j − lr · g_j / m, lr · λ / m)` for `j ≥ 1`. Bias updated without regularization. Does **not** use the shared `optimizer.step()` mechanism — updates are applied manually. Loss is stored in `self.loss_history` (not `optimizer.history`).

`lam` must be non-negative.

---

## Example Session

```python
from Vectors.vector import Vector
from Matrix.matrix import Matrix
from ML_Models.linear.linear_regression import (
    LinearRegression, GDLinearRegression, RidgeRegression, LassoRegression
)
from ML_Models.linear.logistic_regression import (
    LogisticRegression, LogisticRegressionL2, LogisticRegressionL1
)
from ML_Models.metrics import mse, rmse, r2_score, precision, recall, f1_score, roc_auc

# --- Regression data ---
X = Matrix([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
y = Vector([2.5, 4.0, 5.5, 7.0])

# OLS
ols = LinearRegression()
ols.fit(X, y)
print(ols.score(X, y))          # R²
print(rmse(y, ols.predict(X)))  # RMSE

# Gradient descent
gd = GDLinearRegression()
gd.fit(X, y, n_iter=2000, lr=0.01)
print(gd.score(X, y))

# Ridge
ridge = RidgeRegression()
ridge.fit(X, y, lam=0.5)
print(ridge.score(X, y))

# Lasso
lasso = LassoRegression()
lasso.fit(X, y, lam=0.05)
print(lasso.score(X, y))

# --- Classification data ---
X_c = Matrix([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]])
y_c = Vector([0.0, 0.0, 1.0, 1.0])

# Logistic
lr = LogisticRegression(lr=0.1, n_iter=500)
lr.fit(X_c, y_c)
probs  = lr.predict_proba(X_c)
labels = lr.predict(X_c)
print(lr.score(X_c, y_c))              # accuracy
print(f1_score(y_c, labels))
print(roc_auc(y_c, probs))

# L2
lr_l2 = LogisticRegressionL2(lam=0.5, lr=0.1, n_iter=500)
lr_l2.fit(X_c, y_c)
print(lr_l2.score(X_c, y_c))

# L1
lr_l1 = LogisticRegressionL1(lam=0.05, lr=0.1, n_iter=500)
lr_l1.fit(X_c, y_c)
print(lr_l1.loss_history)
```

---

## Error Reference

| Situation | Exception |
|---|---|
| `X` not a `Matrix` or `y` not a `Vector` | `TypeError` |
| `X.n_rows != len(y)` | `ValueError` |
| Empty `X` or `y` | `ValueError` |
| `predict()` or `score()` before `fit()` | `RuntimeError` |
| `predict()` passed a non-`Matrix` | `TypeError` |
| `lr <= 0` | `ValueError` |
| `n_iter < 1` or `n_iters < 1` | `ValueError` |
| `lam < 0` in Ridge, Lasso, L1, L2 | `ValueError` |
| `tol <= 0` in Lasso | `ValueError` |
| `threshold` outside `(0, 1)` in `predict()` | `ValueError` |
| Non-binary values in `y_pred` for classification metrics | `ValueError` |
| Mismatched lengths in any metric | `ValueError` |
| Empty vectors in any metric | `ValueError` |

---

## Design Notes

- **Bias column prepended internally:** `_add_bias_column` always augments `X` before any computation. Weights `self.w[0]` is the bias; feature weights start at index 1. This is consistent across all seven model classes.
- **Bias not regularized in Ridge, Lasso, L1, L2:** The regularization penalty and proximal steps explicitly exclude `w[0]` by iterating from index 1.
- **`GDLinearRegression` vs. `LinearRegression`:** Use OLS for small, well-conditioned datasets where the closed form is exact and fast. Use gradient descent for large datasets or when experimenting with optimizers.
- **`LassoRegression` uses coordinate descent, not gradient descent:** Coordinate descent is exact for L1-penalized least squares and avoids the non-differentiability of `|w|` at zero. `GDLinearRegression` with a subgradient would be an approximation; coordinate descent is not.
- **`LogisticRegressionL1` does not use the `Optimizer` abstraction:** The proximal gradient step (soft-threshold after a gradient step) cannot be expressed as a standard `optimizer.step()` call. The update loop is manual, and losses are stored in `self.loss_history` rather than `optimizer.history`.
- **`roc_auc` handles ties:** Tied probability scores are batched before trapezoidal area accumulation. Naive per-sample traversal over tied scores over-counts partial rectangles.
- **`sigmoid` is numerically stable:** Positive inputs use `1 / (1 + exp(-z))`; negative inputs use `exp(z) / (1 + exp(z))`, avoiding `exp` overflow in both branches.

---

## Roadmap Context

These files are among the first machine learning model implementations in the Engineering Redemption Arc curriculum. They depend on:

- **`Vectors/`** — Project 1
- **`Matrix/`** — Project 2
- **`Probability/`** — Projects 3–7 (via metrics and statistical foundations)
- **`InformationTheory/`** — Project 8 (`binary_cross_entropy` loss)
- **`Optimization/`** — Project 10 (Adam, SGD, and the optimizer abstraction)

They establish the `MLModels` base class and `metrics.py` infrastructure that all subsequent ML model implementations in Phase 2 (Projects 12–25) will inherit and extend.
