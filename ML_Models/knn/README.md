# ML_Models / knn — Pure Python K-Nearest Neighbors

## Overview

The `ML_Models/knn/` sub-package implements K-Nearest Neighbors classification and regression from scratch — part of the *Engineering Redemption Arc*, a structured 60-project ML engineering curriculum in the [`ml-implementation-from-scratch`](https://github.com/) repository.

Two model classes are provided — `KNNClassifier` and `KNNRegressor` — sharing a common `_KNNBase` class that handles fitting, neighbor retrieval, and parameter reporting. Both support standard uniform voting/averaging and inverse-distance-squared weighting. Distance computation is handled by the sibling `distances.py` module, which ships three metrics and a resolver that accepts custom callables. All models accept `Matrix` feature inputs and `Vector` targets. No NumPy, SciPy, or scikit-learn is used.

---

## Project Structure

```
ml-implementation-from-scratch/
├── Vectors/
│   └── vector.py
├── Matrix/
│   └── matrix.py
├── ML_Models/
│   ├── base_class.py              # MLModels abstract base class
│   ├── metrics.py                 # accuracy_score, r2_score
│   └── knn/
│       ├── distances.py           # euclidean, manhattan, minkowski, _resolve_metric
│       └── knn.py                 # _KNNBase, KNNClassifier, KNNRegressor
```

---

## Dependencies

| Requirement | Detail |
|---|---|
| Python | 3.10+ |
| `math`, `os`, `sys`, `typing` | Standard library only |
| `Vectors/vector.py` | Feature rows, label vectors, predictions |
| `Matrix/matrix.py` | Feature matrix input to `fit()` and `predict()` |
| `ML_Models/base_class.py` | `MLModels` abstract base class |
| `ML_Models/metrics.py` | `accuracy_score` (classifier), `r2_score` (regressor) |
| `ML_Models/knn/distances.py` | Distance functions and metric resolver |

No `pip install` required.

---

## Installation

```python
import sys
sys.path.insert(0, "/path/to/ml-implementation-from-scratch")

from ML_Models.knn.distances import euclidean, manhattan, minkowski
from ML_Models.knn.knn import KNNClassifier, KNNRegressor
```

---

## Module Reference

---

### `distances.py` — Distance Functions

Three distance functions and a metric resolver. All distance functions accept two `Vector` arguments of equal length and return a `float`.

---

#### `euclidean(a, b)`

L2 distance: `‖a − b‖₂`.

```python
from ML_Models.knn.distances import euclidean

euclidean(Vector([1.0, 2.0]), Vector([4.0, 6.0]))   # 5.0
```

Delegates to `(a - b).norm(2)`.

---

#### `manhattan(a, b)`

L1 distance: `‖a − b‖₁`.

```python
from ML_Models.knn.distances import manhattan

manhattan(Vector([1.0, 2.0]), Vector([4.0, 6.0]))   # 7.0
```

Delegates to `(a - b).norm(1)`.

---

#### `minkowski(a, b, p)`

General Lp distance: `(Σ |aᵢ − bᵢ|ᵖ)^(1/p)`.

```python
from ML_Models.knn.distances import minkowski

minkowski(Vector([1.0, 2.0]), Vector([4.0, 6.0]), p=3)
```

`p` must be strictly greater than `0`. Note that `minkowski(a, b, p=1)` equals `manhattan` and `minkowski(a, b, p=2)` equals `euclidean`, but the implementation is a direct loop rather than a `Vector.norm()` call.

---

#### `_resolve_metric(metric)`

Resolves a metric string or callable to a distance function. Used internally by `_KNNBase`.

```python
from ML_Models.knn.distances import _resolve_metric

fn = _resolve_metric('euclidean')       # returns euclidean function
fn = _resolve_metric('manhattan')       # returns manhattan function
fn = _resolve_metric(my_dist_fn)        # passes callable through unchanged
```

| Input | Behavior |
|---|---|
| `'euclidean'` | Returns `euclidean` |
| `'manhattan'` | Returns `manhattan` |
| Any other string | Raises `ValueError` with list of valid options |
| Callable | Returned as-is |
| Anything else | Raises `TypeError` |

`minkowski` is not in `_METRIC_MAP` and cannot be selected by string — pass it as a lambda: `lambda a, b: minkowski(a, b, p=3)`.

---

### `knn.py` — KNN Models

---

### `_KNNBase` — Shared Base Class

Internal base class inherited by both `KNNClassifier` and `KNNRegressor`. Not intended to be instantiated directly.

```python
_KNNBase(k=5, metric='euclidean')
```

| Parameter | Default | Constraint |
|---|---|---|
| `k` | `5` | Positive `int` |
| `metric` | `'euclidean'` | `'euclidean'`, `'manhattan'`, or any callable |

**Shared interface:**

```python
model.fit(X, y)         # store training rows and labels; O(1) — lazy learner
model.parameters()      # return {'k': int, 'metric': str, 'n_train': int}
```

**Internal helpers:**

| Method | Purpose |
|---|---|
| `_check_fitted()` | Raises `RuntimeError` if `fit()` has not been called |
| `_get_neighbors(x)` | Returns the `k` nearest `(distance, label)` pairs for a single query vector, sorted ascending by distance |

`_get_neighbors` computes distances to all training points, sorts in O(n log n), and slices the first `k`. No approximate search or KD-tree is used.

---

### `KNNClassifier`

Binary and multi-class classifier. Predicts by majority vote over the `k` nearest neighbors.

```python
from ML_Models.knn.knn import KNNClassifier

model = KNNClassifier(k=5, metric='euclidean')
model.fit(X_train, y_train)

labels = model.predict(X_test)                   # Vector of predicted class labels
labels = model.predict_weighted(X_test)          # Vector — inverse-distance-squared vote
probs  = model.predict_proba(X_test)             # list[dict] — class probabilities
acc    = model.score(X_test, y_test)             # accuracy (float)
```

---

#### `predict(X)`

Majority vote: the label with the highest raw count among the `k` neighbors wins. Ties are broken by whichever label `max()` returns first (insertion order of the count dict).

```python
labels = model.predict(X_test)   # Vector of floats
```

---

#### `predict_weighted(X)`

Inverse-distance-squared vote: each neighbor contributes weight `1 / d²` to its label's total. The label with the highest accumulated weight wins.

```python
labels = model.predict_weighted(X_test)
```

**Exact-match handling:** if any neighbor has distance `0.0`, its weight becomes `inf`. In that case the tie-breaking rule falls back to majority vote among only the exact matches (distance == 0), ignoring all non-zero-distance neighbors.

---

#### `predict_proba(X)`

Returns a `list` of `dict`s, one per query point. Each dict maps class labels to their fraction of the `k` neighbors.

```python
probs = model.predict_proba(X_test)
# [{'0.0': 0.4, '1.0': 0.6}, ...]
```

Probabilities sum to `1.0` for each query point. Only labels present among the `k` neighbors appear in the dict.

---

#### `score(X, y)`

Returns accuracy: fraction of predictions matching `y`.

```python
acc = model.score(X_test, y_test)   # float in [0.0, 1.0]
```

Delegates to `accuracy_score(y, self.predict(X))`.

---

### `KNNRegressor`

Regression by averaging the labels of the `k` nearest neighbors.

```python
from ML_Models.knn.knn import KNNRegressor

model = KNNRegressor(k=5, metric='euclidean')
model.fit(X_train, y_train)

preds = model.predict(X_test)            # Vector of mean predictions
preds = model.predict_weighted(X_test)   # Vector — inverse-distance-squared average
r2    = model.score(X_test, y_test)      # R² (float)
```

---

#### `predict(X)`

Uniform mean: `ŷ = (1/k) Σ yᵢ` over the `k` nearest neighbors.

```python
preds = model.predict(X_test)   # Vector of floats
```

---

#### `predict_weighted(X)`

Inverse-distance-squared weighted average: `ŷ = Σ (yᵢ / dᵢ²) / Σ (1 / dᵢ²)`.

```python
preds = model.predict_weighted(X_test)
```

**Exact-match handling:** if any neighbor has distance `0.0`, the prediction is the unweighted mean of all exact-match labels, ignoring non-zero-distance neighbors entirely.

---

#### `score(X, y)`

Returns the coefficient of determination R².

```python
r2 = model.score(X_test, y_test)   # float; 1.0 = perfect, can be negative
```

Delegates to `r2_score(y, self.predict(X))`.

---

## Example Session

```python
from Vectors.vector import Vector
from Matrix.matrix import Matrix
from ML_Models.knn.knn import KNNClassifier, KNNRegressor
from ML_Models.knn.distances import euclidean, manhattan, minkowski
from ML_Models.metrics import accuracy_score, r2_score

# --- Classification ---
X_train = Matrix([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0]])
y_train = Vector([0.0, 0.0, 1.0, 1.0, 1.0])

X_test = Matrix([[2.5, 2.5], [4.0, 1.5]])
y_test = Vector([0.0, 1.0])

clf = KNNClassifier(k=3, metric='euclidean')
clf.fit(X_train, y_train)

labels  = clf.predict(X_test)
wlabels = clf.predict_weighted(X_test)
probs   = clf.predict_proba(X_test)
acc     = clf.score(X_test, y_test)

print(labels)                    # Vector of predicted labels
print(probs)                     # [{'0.0': ..., '1.0': ...}, ...]
print(acc)                       # accuracy

# Manhattan metric
clf_mh = KNNClassifier(k=3, metric='manhattan')
clf_mh.fit(X_train, y_train)
print(clf_mh.score(X_test, y_test))

# Custom metric (Minkowski p=3)
clf_mk = KNNClassifier(k=3, metric=lambda a, b: minkowski(a, b, p=3))
clf_mk.fit(X_train, y_train)
print(clf_mk.predict(X_test))

# --- Regression ---
X_r = Matrix([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_r = Vector([1.5, 2.5, 3.5, 4.5, 5.5])

X_r_test = Matrix([[2.5], [3.5]])
y_r_test  = Vector([3.0, 4.0])

reg = KNNRegressor(k=3, metric='euclidean')
reg.fit(X_r, y_r)

preds  = reg.predict(X_r_test)
wpreds = reg.predict_weighted(X_r_test)
r2     = reg.score(X_r_test, y_r_test)

print(preds)     # Vector of mean predictions
print(wpreds)    # Vector of weighted predictions
print(r2)        # R²

# parameters()
print(clf.parameters())    # {'k': 3, 'metric': 'euclidean', 'n_train': 5}
print(reg.parameters())    # {'k': 3, 'metric': 'euclidean', 'n_train': 5}
```

---

## Error Reference

| Situation | Exception |
|---|---|
| `k` not a positive `int` | `ValueError` |
| Unknown metric string | `ValueError` |
| `metric` not a string or callable | `TypeError` |
| `X.n_rows != len(y)` in `fit()` | `ValueError` |
| Empty training set in `fit()` | `ValueError` |
| `predict()` / `score()` before `fit()` | `RuntimeError` |
| Vectors of unequal length in any distance function | `ValueError` |
| `p <= 0` in `minkowski` | `ValueError` |

---

## Design Notes

- **Lazy learner — `fit()` is O(1):** KNN stores training rows as a `list[Vector]` and the label `Vector` directly. No model parameters are computed at training time. All computation is deferred to `predict()`.
- **Prediction is O(n · d) per query point:** For each test point, distances to all `n` training points are computed (O(n · d)), then sorted (O(n log n)). No KD-tree, ball-tree, or approximate indexing is used — exact brute-force search.
- **`_get_neighbors` returns `(distance, label)` tuples:** Both voting and averaging methods operate on this shared representation. The sort is stable — equal distances preserve insertion order.
- **`predict_weighted` and `predict` give identical results when all distances are equal:** Uniform weights collapse to the majority/mean rule.
- **Exact-match (distance = 0) edge case:** Inverse-distance weights blow up to `inf` at zero distance. Both classifier and regressor handle this explicitly: the prediction reduces to majority vote or mean over only the exact-match neighbors.
- **`predict_proba` is count-based, not distance-weighted:** Probabilities reflect the raw fraction of the `k` neighbors belonging to each class. For distance-weighted probabilities, use `predict_weighted` and interpret the result as the most probable class.
- **`_metric_name` stores `'custom'` for callable metrics:** `parameters()` cannot recover the name of a user-supplied callable, so it falls back to `'custom'`.
- **Multi-class support in `KNNClassifier`:** Majority vote and weighted vote both work for any number of classes — labels are treated as arbitrary hashable floats.

---

## Roadmap Context

This sub-package is among the first model implementations in Phase 2 of the Engineering Redemption Arc curriculum. It depends on:

- **`Vectors/`** — Project 1
- **`Matrix/`** — Project 2
- **`ML_Models/base_class.py`** and **`ML_Models/metrics.py`** — established by the `linear/` sub-package (Project 11)

KNN is a non-parametric baseline that requires no optimization and no `InformationTheory/` dependency, making it a self-contained reference point for comparing against the parametric models (linear, logistic, SVM, trees) built in subsequent projects.
