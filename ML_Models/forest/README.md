# ML_Models / trees — Pure Python Decision Trees

## Overview

The `ML_Models/trees/` sub-package implements Decision Tree classification and regression from scratch — part of the *Engineering Redemption Arc*, a structured 60-project ML engineering curriculum in the [`ml-implementation-from-scratch`](https://github.com/) repository.

Two model classes are provided — `DecisionTreeClassifier` and `DecisionTreeRegressor` — built on a shared recursive tree construction engine. The classifier supports Gini impurity and information gain (entropy) as splitting criteria; the regressor uses variance reduction. Both models accept `Matrix` feature inputs and `Vector` targets, integrate with `InformationTheory/` for impurity computation, and expose tree introspection via `get_depth()` and `get_n_leaves()`. No NumPy, SciPy, or scikit-learn is used.

---

## Project Structure

```
ml-implementation-from-scratch/
├── Vectors/
│   └── vector.py
├── Matrix/
│   └── matrix.py
├── InformationTheory/
│   └── information_theory.py      # information_gain, gini_gain
└── ML_Models/
    ├── base_class.py              # MLModels abstract base class
    ├── metrics.py                 # accuracy, r2_score
    └── trees/
        └── decision_tree.py       # Node, DecisionTreeClassifier, DecisionTreeRegressor
```

---

## Dependencies

| Requirement | Detail |
|---|---|
| Python | 3.10+ |
| `os`, `sys`, `typing`, `collections.Counter` | Standard library only |
| `Vectors/vector.py` | Target vectors, predictions |
| `Matrix/matrix.py` | Feature matrix input to `fit()` and `predict()` |
| `InformationTheory/information_theory.py` | `information_gain`, `gini_gain` |
| `ML_Models/base_class.py` | `MLModels` abstract base class |
| `ML_Models/metrics.py` | `accuracy` (classifier), `r2_score` (regressor) |

No `pip install` required.

---

## Installation

```python
import sys
sys.path.insert(0, "/path/to/ml-implementation-from-scratch")

from ML_Models.trees.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
```

---

## Internal Architecture

---

### `Node`

The tree is composed of `Node` objects linked recursively. Not intended for direct instantiation — created internally by `_build`.

```python
node.feature_idx   # int — feature to split on (None at leaves)
node.threshold     # float — split threshold (None at leaves)
node.left          # Node — subtree for X[feature] <= threshold
node.right         # Node — subtree for X[feature] > threshold
node.value         # float — prediction value (set only at leaves)

node.is_leaf()     # True if value is not None
repr(node)         # "Leaf(value=1.0)" or "Node(feature=2, threshold=3.1416)"
```

A node is an internal split node when `value is None` and a leaf when `value` is set. After `fit()`, `model.parameters()` returns `[self._root]` — the root `Node` of the fitted tree.

---

### Splitting Engine (module-level helpers)

These functions implement the CART-style greedy search used by both classifiers and regressors.

**`_best_split(X_data, y, criterion)`** — exhaustive search over all features and candidate thresholds.

For each feature `j`, candidate thresholds are the midpoints between consecutive unique sorted values (via `_unique_thresholds`). For each `(feature, threshold)` pair, the gain function for the chosen criterion is evaluated and the globally best split is returned as `{"gain": float, "feature_idx": int, "threshold": float}`. Splits that produce an empty left or right child are skipped.

**Gain functions by criterion:**

| Criterion | Used by | Formula |
|---|---|---|
| `"gini"` | `DecisionTreeClassifier` | `gini_gain` from `InformationTheory/` |
| `"entropy"` | `DecisionTreeClassifier` | `information_gain` from `InformationTheory/` |
| `"variance"` | `DecisionTreeRegressor` | `Var(y) − (nₗ/n)·Var(yₗ) − (nᵣ/n)·Var(yᵣ)` |

Both `information_gain` and `gini_gain` receive class counts (not raw labels): `counts(y)` is built via `Counter` over the sorted unique classes in `y`, so the same class ordering is used for parent and both children.

**`_build(X_data, y, criterion, leaf_fn, depth, max_depth, min_samples_split, min_impurity_decrease)`** — recursive tree construction.

Stops and creates a leaf when any of the following hold:
- All labels in `y` are identical (pure node)
- `len(y) < min_samples_split`
- `depth >= max_depth` (when `max_depth` is not `None`)
- No valid split exists (`feature_idx is None`) or the best gain is below `min_impurity_decrease`

Otherwise, splits on the best feature/threshold and recurses into left and right subtrees at `depth + 1`.

**Leaf value functions:**

| Model | Function | Value |
|---|---|---|
| `DecisionTreeClassifier` | `_leaf_value_classifier` | Most common label in `y` (majority vote via `Counter.most_common`) |
| `DecisionTreeRegressor` | `_leaf_value_regressor` | Mean of `y` |

Both raise `ValueError` on empty `y`.

---

## Model Reference

---

### `DecisionTreeClassifier`

Multi-class classifier. Splits on Gini impurity decrease or information gain; predicts by majority vote at leaves.

```python
from ML_Models.trees.decision_tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion="gini",        # 'gini' or 'entropy'
    max_depth=None,          # int >= 1 or None (unlimited)
    min_samples_split=2,     # int >= 2
    min_impurity_decrease=0.0
)
clf.fit(X_train, y_train)

labels = clf.predict(X_test)      # Vector of predicted class labels (float)
acc    = clf.score(X_test, y_test) # accuracy (float)
root   = clf.parameters()          # [Node] — root of the fitted tree
depth  = clf.get_depth()           # int — depth of the fitted tree
leaves = clf.get_n_leaves()        # int — number of leaf nodes
```

| Parameter | Default | Constraint |
|---|---|---|
| `criterion` | `'gini'` | `'gini'` or `'entropy'` |
| `max_depth` | `None` | `int >= 1` or `None` |
| `min_samples_split` | `2` | `int >= 2` |
| `min_impurity_decrease` | `0.0` | `float >= 0.0` |

**`fit(X, y)`**

Converts `X` to a `list[list[float]]` and `y` to a `list` internally, then calls `_build` with the chosen criterion and `_leaf_value_classifier`. Sets `self._root` to the returned root `Node`. Labels may be any numeric values — multi-class is supported.

**`predict(X)`**

Traverses the tree for each row of `X` via `_predict_one`. At each internal node, routes left if `row[feature_idx] <= threshold`, right otherwise. Returns a `Vector` of leaf values (majority-vote class labels as `float`).

**`score(X, y)`**

Returns accuracy: fraction of predictions matching `y`. Delegates to `accuracy(y, self.predict(X))`.

**`get_depth()`**

Returns the depth of the fitted tree as an `int`. A single-leaf tree (root is a leaf) has depth `0`. Computed by `_tree_depth` — a recursive post-order traversal.

**`get_n_leaves()`**

Returns the total number of leaf nodes as an `int`. Computed by `_count_leaves` — a recursive traversal that counts nodes where `is_leaf()` is `True`.

---

### `DecisionTreeRegressor`

Regression tree. Splits by variance reduction; predicts as the mean of training labels at each leaf.

```python
from ML_Models.trees.decision_tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(
    max_depth=None,
    min_samples_split=2,
    min_impurity_decrease=0.0
)
reg.fit(X_train, y_train)

preds  = reg.predict(X_test)       # Vector of mean predictions (float)
r2     = reg.score(X_test, y_test) # R² (float)
root   = reg.parameters()           # [Node] — root of the fitted tree
depth  = reg.get_depth()            # int
leaves = reg.get_n_leaves()         # int
```

| Parameter | Default | Constraint |
|---|---|---|
| `max_depth` | `None` | `int >= 1` or `None` |
| `min_samples_split` | `2` | `int >= 2` |
| `min_impurity_decrease` | `0.0` | `float >= 0.0` |

`DecisionTreeRegressor` has no `criterion` parameter — variance reduction is the only splitting criterion and is hardcoded in `_build`.

**`fit(X, y)`**

Identical flow to `DecisionTreeClassifier.fit`, using `"variance"` as the criterion and `_leaf_value_regressor` as the leaf function.

**`predict(X)`**

Same traversal logic as the classifier. Leaf values are means of training labels in that partition.

**`score(X, y)`**

Returns R². Delegates to `r2_score(y, self.predict(X))`. Can be negative for poorly fitted trees.

**`get_depth()` / `get_n_leaves()`**

Same as classifier — shared `_tree_depth` and `_count_leaves` helpers.

---

## Example Session

```python
from Vectors.vector import Vector
from Matrix.matrix import Matrix
from ML_Models.trees.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from ML_Models.metrics import accuracy, r2_score

# --- Classification ---
X_train = Matrix([
    [2.0, 3.0], [1.0, 1.0], [3.0, 2.0],
    [6.0, 5.0], [7.0, 8.0], [8.0, 6.0]
])
y_train = Vector([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

X_test = Matrix([[2.5, 2.5], [7.0, 7.0]])
y_test = Vector([0.0, 1.0])

# Gini (default)
clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
clf.fit(X_train, y_train)

labels = clf.predict(X_test)
acc    = clf.score(X_test, y_test)
print(labels)                     # Vector([0.0, 1.0])
print(acc)                        # 1.0
print(clf.get_depth())            # int
print(clf.get_n_leaves())         # int
print(clf.parameters())           # [Node(...)]

# Entropy criterion
clf_e = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=3)
clf_e.fit(X_train, y_train)
print(clf_e.score(X_test, y_test))

# Pruning via min_impurity_decrease
clf_p = DecisionTreeClassifier(criterion="gini", min_impurity_decrease=0.01)
clf_p.fit(X_train, y_train)
print(clf_p.get_depth())

# --- Regression ---
X_r = Matrix([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y_r = Vector([1.2, 1.9, 3.1, 3.8, 5.2, 6.0])

X_r_test = Matrix([[2.5], [4.5]])
y_r_test  = Vector([2.5, 4.5])

reg = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
reg.fit(X_r, y_r)

preds = reg.predict(X_r_test)
r2    = reg.score(X_r_test, y_r_test)
print(preds)                      # Vector of mean predictions
print(r2)                         # R²
print(reg.get_depth())
print(reg.get_n_leaves())

# Shallow tree (high bias, low variance)
reg_shallow = DecisionTreeRegressor(max_depth=1)
reg_shallow.fit(X_r, y_r)
print(reg_shallow.get_n_leaves()) # 2
```

---

## Error Reference

| Situation | Exception |
|---|---|
| `criterion` not `'gini'` or `'entropy'` | `ValueError` |
| `max_depth < 1` (when not `None`) | `ValueError` |
| `min_samples_split < 2` | `ValueError` |
| `min_impurity_decrease < 0.0` | `ValueError` |
| `X` not a `Matrix` or `y` not a `Vector` | `TypeError` |
| `X.n_rows != len(y)` | `ValueError` |
| Empty `X` or `y` | `ValueError` |
| `predict()`, `score()`, `parameters()`, `get_depth()`, or `get_n_leaves()` before `fit()` | `RuntimeError` |
| `X` not a `Matrix` in `predict()` | `TypeError` |
| `_leaf_value_classifier` or `_leaf_value_regressor` called on empty node | `ValueError` |

---

## Design Notes

- **CART-style greedy splitting:** At each node, all features and all midpoint thresholds between consecutive unique values are evaluated exhaustively. The split maximizing the chosen gain function is selected. No random feature subsampling is applied — this is a deterministic, full-feature search. (Feature subsampling is added at the `RandomForest` level in `forest/`.)
- **Midpoint thresholds avoid data leakage:** Thresholds are set to the midpoint between adjacent unique feature values, not to the values themselves. This prevents the threshold from coinciding with a training point and ensures left/right splits are unambiguous.
- **`InformationTheory/` receives class counts, not raw labels:** `_information_gain` and `_gini_gain` build count vectors over the sorted unique classes in the full `y` before passing to `information_gain` and `gini_gain`. This guarantees the same class ordering is used for parent and both children, which is required for the information-theoretic gain formulas to be consistent.
- **Variance reduction is self-contained:** Unlike the classifier criteria, `_variance_reduction` is implemented directly in `decision_tree.py` and does not call into `InformationTheory/`. It follows the standard formula: `Var(y) − (nₗ/n)·Var(yₗ) − (nᵣ/n)·Var(yᵣ)`.
- **`parameters()` returns the root `Node`:** Unlike parametric models where `parameters()` returns weight vectors, the tree's learned structure is the root node of the recursively linked `Node` graph. The full tree is accessible by traversing from `self._root`.
- **`self.w = True` sentinel for `_check_is_fitted`:** `MLModels._check_is_fitted` checks for the existence of `self.w`. Since decision trees have no weight vector, `fit()` sets `self.w = True` as a sentinel value. The actual fitted state check in both classes uses `self._root is None` directly.
- **`get_depth()` returns 0 for a single-leaf tree:** `_tree_depth` returns `0` when called on a leaf node. A tree where all training labels are identical — so `_build` immediately returns a leaf at depth 0 — has depth `0` and `1` leaf.
- **Multi-class classification supported:** `_leaf_value_classifier` uses `Counter.most_common(1)` over whatever labels appear in the leaf's partition. Labels are stored and returned as `float` but can represent any number of classes.

---

## Roadmap Context

This sub-package depends on:

- **`Vectors/`** — Project 1
- **`Matrix/`** — Project 2
- **`InformationTheory/`** — Project 8 (`information_gain`, `gini_gain`)
- **`ML_Models/base_class.py`** and **`ML_Models/metrics.py`** — established by the `linear/` sub-package (Project 11)

Decision trees are the foundational component for ensemble methods in the next stage of the curriculum. `DecisionTreeClassifier` and `DecisionTreeRegressor` are used directly as the base learners in `ML_Models/forest/random_forest.py`, where feature subsampling and bootstrap aggregation are added on top of the same `_build` engine.
