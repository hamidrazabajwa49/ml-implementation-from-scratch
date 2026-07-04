# InformationTheory — Pure Python Information Theory Library

## Overview

`information_theory.py` is a pure Python implementation of core information-theoretic quantities — built from scratch as part of the *Engineering Redemption Arc*, a structured 60-project ML engineering curriculum in the [`ml-implementation-from-scratch`](https://github.com/) repository.

The module provides standalone functions covering entropy, divergence measures, mutual information, cross-entropy loss, decision tree splitting criteria, and language model perplexity. No third-party libraries are used — the only dependency is Python's standard `math` module.

---

## Project Structure

```
ml-implementation-from-scratch/
└── InformationTheory/
    └── information_theory.py    # All functions — no class wrappers
```

The module is fully self-contained. There are no local imports and no required sibling packages.

---

## Dependencies

| Requirement | Detail |
|---|---|
| Python | 3.8+ |
| `math` (stdlib) | any |

No `pip install` required.

---

## Installation

Copy `information_theory.py` into your project, or import it directly:

```python
import sys
sys.path.insert(0, "/path/to/ml-implementation-from-scratch")

from InformationTheory.information_theory import (
    entropy, kl_divergence, mutual_information,
    cross_entropy, binary_cross_entropy, information_gain
)
```

---

## Conventions

### Probability distributions

Functions that accept a distribution `p` or `q` expect a `list` of non-negative floats that sum to `1.0` (tolerance `1e-6`). Violations raise `ValueError`.

### Joint distributions

Functions that accept a `joint` distribution expect a `list[list[float]]` — a 2D table where `joint[i][j]` = P(X=i, Y=j). All rows must have equal length and all values must be non-negative. The table need not be explicitly validated to sum to 1 before passing to joint functions, but marginals derived from it must be valid distributions.

### Log base

All entropy and divergence functions accept an optional `base` parameter (default `2.0`, yielding bits). Use `base=math.e` for nats or `base=10` for hartleys. Any positive value other than `1.0` is valid.

---

## Function Reference

---

### Entropy

#### `entropy(p, base=2.0)`

Shannon entropy of a discrete distribution.

```python
from InformationTheory.information_theory import entropy

entropy([0.5, 0.5])           # 1.0 bit (maximum for binary)
entropy([1.0, 0.0])           # 0.0 bits (degenerate)
entropy([0.25, 0.25, 0.25, 0.25])  # 2.0 bits
```

Zero-probability terms are skipped (0 · log 0 = 0 by convention).

---

#### `binary_entropy(p, base=2.0)`

Entropy of a Bernoulli(p) distribution. Equivalent to `entropy([p, 1-p])` but accepts a single scalar.

```python
from InformationTheory.information_theory import binary_entropy

binary_entropy(0.5)    # 1.0
binary_entropy(0.0)    # 0.0
binary_entropy(0.9)    # ~0.469
```

`p` must be in `[0, 1]`. Returns `0.0` at the boundary values exactly.

---

#### `joint_entropy(joint, base=2.0)`

Entropy of a joint distribution H(X, Y).

```python
from InformationTheory.information_theory import joint_entropy

joint = [[0.1, 0.4],
         [0.2, 0.3]]

joint_entropy(joint)   # H(X, Y)
```

Internally flattens the 2D table and delegates to `entropy()`.

---

#### `perplexity(p, base=2.0)`

Perplexity of a distribution: `base ** entropy(p)`. Measures the effective alphabet size of the distribution; lower is better for language models.

```python
from InformationTheory.information_theory import perplexity

perplexity([0.5, 0.5])               # 2.0
perplexity([0.25, 0.25, 0.25, 0.25]) # 4.0
```

---

#### `renyi_entropy(p, alpha, base=2.0)`

Rényi entropy of order `alpha`. Generalizes Shannon entropy:

- `alpha → 1`: converges to Shannon entropy (handled analytically)
- `alpha = 0`: Hartley entropy (log of support size)
- `alpha = 2`: collision entropy
- `alpha → ∞`: min-entropy

```python
from InformationTheory.information_theory import renyi_entropy

renyi_entropy([0.5, 0.3, 0.2], alpha=2.0)
renyi_entropy([0.5, 0.3, 0.2], alpha=1.0)   # = entropy([0.5, 0.3, 0.2])
```

`alpha` must be positive. The `alpha == 1` case is detected within `1e-10` tolerance.

---

### Marginals and Conditional Entropy

#### `marginal_from_joint(joint, axis=0)`

Computes marginal distributions from a joint table.

```python
from InformationTheory.information_theory import marginal_from_joint

joint = [[0.1, 0.4],
         [0.2, 0.3]]

marginal_from_joint(joint, axis=0)   # P(Y): sum over rows → [0.3, 0.7]
marginal_from_joint(joint, axis=1)   # P(X): sum over cols → [0.5, 0.5]
```

`axis=0` sums down columns (marginalizes over X, returns P(Y)). `axis=1` sums across rows (marginalizes over Y, returns P(X)).

---

#### `conditional_entropy(joint, given="X", base=2.0)`

Conditional entropy H(Y|X) or H(X|Y), computed via the chain rule: H(Y|X) = H(X,Y) − H(X).

```python
from InformationTheory.information_theory import conditional_entropy

conditional_entropy(joint, given="X")   # H(Y|X)
conditional_entropy(joint, given="Y")   # H(X|Y)
```

Result is clamped to `max(0.0, ...)` to prevent negative values from floating-point rounding.

---

### Mutual Information

#### `mutual_information(joint, base=2.0)`

Mutual information I(X; Y) = H(X) + H(Y) − H(X, Y).

```python
from InformationTheory.information_theory import mutual_information

joint = [[0.1, 0.4],
         [0.2, 0.3]]

mutual_information(joint)   # float >= 0
```

Returns `0.0` for independent variables. Clamped to `max(0.0, ...)`.

---

#### `normalized_mutual_information(joint, base=2.0)`

NMI = I(X;Y) / (0.5 × (H(X) + H(Y))). Scales mutual information to `[0, 1]`, enabling comparison across variables with different marginal entropies.

```python
from InformationTheory.information_theory import normalized_mutual_information

normalized_mutual_information(joint)   # float in [0, 1]
```

Returns `0.0` if both marginals have zero entropy.

---

### Divergence Measures

#### `kl_divergence(p, q, base=2.0)`

Kullback-Leibler divergence D_KL(P ‖ Q). Measures how much P diverges from a reference distribution Q. Not symmetric.

```python
from InformationTheory.information_theory import kl_divergence

kl_divergence([0.5, 0.5], [0.9, 0.1])   # D_KL(P||Q)
kl_divergence([0.5, 0.5], [0.5, 0.5])   # 0.0
```

Returns `float("inf")` when `q[i] == 0` and `p[i] > 0`. Zero-probability terms in `p` are skipped. Result is clamped to `max(0.0, ...)`.

---

#### `js_divergence(p, q, base=2.0)`

Jensen-Shannon divergence — a symmetric, bounded smoothing of KL divergence. JS(P‖Q) = 0.5 · KL(P‖M) + 0.5 · KL(Q‖M), where M = (P+Q)/2.

```python
from InformationTheory.information_theory import js_divergence

js_divergence([0.5, 0.5], [0.9, 0.1])   # float in [0, 1] for base=2
js_divergence(p, q) == js_divergence(q, p)  # always True
```

Always finite (M is never zero where P or Q is nonzero). Range is `[0, 1]` in bits.

---

### Cross-Entropy

#### `cross_entropy(p, q, base=2.0)`

Cross-entropy H(P, Q) = −Σ p_i log q_i. Measures the expected code length when using distribution Q to encode samples from P.

```python
from InformationTheory.information_theory import cross_entropy

cross_entropy([0.5, 0.5], [0.9, 0.1])   # float >= entropy(p)
cross_entropy(p, p)                      # = entropy(p)
```

Returns `float("inf")` when `q[i] == 0` and `p[i] > 0`.

---

#### `binary_cross_entropy(y_true, y_pred_proba, eps=1e-15)`

Binary cross-entropy loss averaged over a dataset — the standard loss function for binary classifiers.

```python
from InformationTheory.information_theory import binary_cross_entropy

y_true = [1, 0, 1, 1, 0]
y_pred = [0.9, 0.1, 0.8, 0.7, 0.2]

loss = binary_cross_entropy(y_true, y_pred)   # float >= 0
```

`y_true` must contain only `0` or `1` values. Predictions are clipped to `[eps, 1-eps]` before taking the log — no explicit validation that predictions are in `[0, 1]` is performed, but extreme values are handled safely. Result is computed in nats (base `e`); there is no `base` parameter for this function.

---

### Decision Tree Criteria

#### `information_gain(parent, subsets, base=2.0)`

Information gain from splitting a node — the primary splitting criterion for ID3 and C4.5 decision trees.

```python
from InformationTheory.information_theory import information_gain

parent  = [30, 70]              # 30 class-A, 70 class-B at parent node
left    = [25, 10]              # left child after split
right   = [5,  60]              # right child after split

ig = information_gain(parent, [left, right])   # float >= 0
```

`parent` and each subset are **count vectors** (not probabilities). Subsets with zero total count are skipped. Result is clamped to `max(0.0, ...)`.

---

#### `gini_impurity(p)`

Gini impurity of a probability distribution: 1 − Σ p_i².

```python
from InformationTheory.information_theory import gini_impurity

gini_impurity([0.5, 0.5])       # 0.5 (maximum for binary)
gini_impurity([1.0, 0.0])       # 0.0 (pure node)
gini_impurity([0.3, 0.3, 0.4])  # ~0.66
```

---

#### `gini_gain(parent, subsets)`

Reduction in Gini impurity from a split — the splitting criterion used by CART decision trees.

```python
from InformationTheory.information_theory import gini_gain

parent = [50, 50]
left   = [45, 5]
right  = [5,  45]

gg = gini_gain(parent, [left, right])   # float >= 0
```

Same count-vector convention as `information_gain`. Subsets with zero total are skipped. Result is clamped to `max(0.0, ...)`.

---

## Example Session

```python
import math
from InformationTheory.information_theory import (
    entropy, binary_entropy, joint_entropy,
    conditional_entropy, mutual_information, normalized_mutual_information,
    cross_entropy, binary_cross_entropy,
    kl_divergence, js_divergence,
    information_gain, gini_impurity, gini_gain,
    perplexity, renyi_entropy,
    marginal_from_joint
)

# --- Entropy ---
p = [0.25, 0.25, 0.25, 0.25]
print(entropy(p))                       # 2.0
print(binary_entropy(0.3))              # ~0.881
print(perplexity(p))                    # 4.0

# --- Joint and conditional ---
joint = [[0.1, 0.4],
         [0.2, 0.3]]
print(joint_entropy(joint))             # H(X, Y)
print(conditional_entropy(joint, given="X"))  # H(Y|X)
print(mutual_information(joint))        # I(X; Y)
print(normalized_mutual_information(joint))   # NMI in [0, 1]

# --- Divergences ---
p2 = [0.6, 0.4]
q2 = [0.5, 0.5]
print(kl_divergence(p2, q2))           # D_KL(P||Q)
print(js_divergence(p2, q2))           # symmetric

# --- Cross-entropy loss ---
print(cross_entropy(p2, q2))
print(binary_cross_entropy([1,0,1], [0.9, 0.1, 0.8]))

# --- Decision tree criteria ---
parent = [30, 70]
left   = [25, 10]
right  = [5,  60]
print(information_gain(parent, [left, right]))
print(gini_gain(parent, [left, right]))

# --- Rényi entropy ---
print(renyi_entropy([0.5, 0.3, 0.2], alpha=2.0))
print(renyi_entropy([0.5, 0.3, 0.2], alpha=1.0))  # = Shannon entropy
```

---

## Error Reference

| Situation | Exception |
|---|---|
| `base <= 0`, `base == 1`, or non-numeric `base` | `ValueError` |
| Empty distribution | `ValueError` |
| Negative probability in distribution | `ValueError` |
| Distribution does not sum to 1 (tolerance `1e-6`) | `ValueError` |
| `p` outside `[0, 1]` in `binary_entropy` | `ValueError` |
| Mismatched lengths in `cross_entropy` / `kl_divergence` / `js_divergence` | `ValueError` |
| `q[i] == 0` where `p[i] > 0` in `cross_entropy` / `kl_divergence` | Returns `float("inf")` |
| Non-binary values in `y_true` | `ValueError` |
| Mismatched lengths in `binary_cross_entropy` | `ValueError` |
| Empty `joint` table or zero-length rows | `ValueError` |
| Jagged (non-rectangular) `joint` table | `ValueError` |
| `axis` not 0 or 1 in `marginal_from_joint` | `ValueError` |
| `given` not `"X"` or `"Y"` in `conditional_entropy` | `ValueError` |
| Empty or all-zero `parent` in `information_gain` / `gini_gain` | `ValueError` |
| Negative counts in `parent` or `subsets` | `ValueError` |
| `alpha <= 0` in `renyi_entropy` | `ValueError` |

---

## Design Notes

- **No class wrappers:** All functions are module-level. The file has no state; every call is pure and side-effect-free.
- **Shared validation helpers:** `_validate_base`, `_validate_distribution`, and `_validate_joint` are private (underscore-prefixed) and called at the top of every public function. They are not part of the public API.
- **Count vectors vs. probability vectors:** `information_gain` and `gini_gain` accept raw class counts rather than normalized probabilities. Normalization happens internally, so passing counts directly from a dataset is correct.
- **`binary_cross_entropy` uses nats:** Unlike all other functions, `binary_cross_entropy` does not expose a `base` parameter — it always uses `math.log` (natural log). This matches the convention of most ML frameworks and makes values directly comparable to PyTorch/sklearn's `binary_cross_entropy` output.
- **Numerical floor on divergences:** `kl_divergence`, `js_divergence`, `mutual_information`, `conditional_entropy`, `information_gain`, and `gini_gain` all clamp results to `max(0.0, ...)` to suppress spurious negative values from floating-point rounding.
- **`renyi_entropy` at `alpha=1`:** The limit as α→1 of Rényi entropy is Shannon entropy. Rather than taking the limit analytically, the code detects `|alpha - 1| < 1e-10` and delegates directly to `entropy()`.

---

## Roadmap Context

This module is a foundational project in the Engineering Redemption Arc curriculum. It has no local dependencies — only `math` from the standard library.

It underpins:

- **Decision tree implementations** (Phase 2) — `information_gain` and `gini_gain` are the direct splitting criteria used in ID3, C4.5, and CART.
- **Neural network training** — `binary_cross_entropy` is the loss function for binary classifiers; `cross_entropy` generalizes to multiclass.
- **Probabilistic model evaluation** — `kl_divergence` and `js_divergence` measure distributional shift in generative models and variational inference.
- **Feature selection and clustering** — `mutual_information` and `normalized_mutual_information` are standard metrics for evaluating feature relevance and cluster quality.
