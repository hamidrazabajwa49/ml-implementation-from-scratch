# InformationTheory — Entropy, Divergences & Tree-Splitting Criteria (From Scratch)

Part of the [`ml-implementation-from-scratch`](../../) curriculum —
**Phase 1: Math Foundations**.

A single-file, dependency-free library of information-theoretic
quantities used throughout ML: Shannon entropy and its variants (binary,
joint, conditional), mutual information, cross entropy, KL/Jensen-Shannon
divergence, decision-tree splitting criteria (information gain, Gini
gain), perplexity, and Renyi entropy. NumPy/SciPy are used only in the
test suite, as correctness oracles.

---

## Overview

All functions operate on plain Python lists/sequences of probabilities
(or, for the tree-splitting criteria, raw non-negative counts) — there
are no external dependencies. Every "log" is base-configurable (default
base 2, i.e. bits) via a `base` parameter, dispatched internally to the
most numerically precise stdlib routine available for common bases (2,
e, 10) rather than the generic two-argument `math.log(x, base)`.

| Category | Functions |
|---|---|
| Shannon entropy family | `entropy`, `binary_entropy`, `joint_entropy` |
| Joint-distribution quantities | `marginal_from_joint`, `conditional_entropy`, `mutual_information`, `normalized_mutual_information` |
| Cross entropy / divergences | `cross_entropy`, `binary_cross_entropy`, `kl_divergence`, `js_divergence` |
| Decision-tree splitting criteria | `information_gain`, `gini_impurity`, `gini_gain` |
| Perplexity / generalized entropy | `perplexity`, `renyi_entropy` |

---

## Project Structure

```
math_foundations/
└── InformationTheory/
    ├── information_theory.py
    ├── tests/
    │   └── test_information_theory.py
    └── README.md
```

This module is self-contained (stdlib `math`/`logging`/`typing` only) —
it does not import from `Vectors`, `Matrix`, `Probability`, `Bayesian`,
or `Optimization`.

---

## Dependencies

| Component | Requires |
|---|---|
| `information_theory.py` | Python 3.8+ standard library only (`math`, `logging`, `typing`) |
| `tests/test_information_theory.py` | `pytest`, `numpy`, `scipy` (test-only, for regression checks) |

Install test dependencies:

```bash
pip install pytest numpy scipy
```

Run the suite from `math_foundations/InformationTheory/`:

```bash
pytest tests/test_information_theory.py -v
```

---

## Module Reference

### Validation helpers (internal)

`_validate_base`, `_validate_distribution`, `_validate_joint`, `_validate_counts` — all reject `bool` (`TypeError`) and explicitly check for NaN/Inf rather than relying on comparison propagation (a naive `x < 0.0` non-negativity check silently lets NaN through, since any comparison against NaN is `False`). `_validate_distribution` requires the sequence to sum to `1.0` within a configurable `tol` (default `1e-6`); `_validate_joint` checks rectangularity and non-negativity but deliberately does *not* itself enforce summing to 1 — callers treating the table as probabilities get that check for free via a downstream `entropy` call, while callers treating it as raw counts aren't forced into an inapplicable constraint.

### Shannon entropy family

```python
entropy(p: Sequence[Number], base: float = 2.0, tol: float = 1e-6) -> float
```
`H(p) = -sum(p_i * log_base(p_i))`. Terms with `p_i == 0` are skipped (by convention, `0 log 0 = 0`). Raises `TypeError` for non-numeric/boolean elements or base, `ValueError` for negative/NaN/Inf probabilities, a distribution not summing to 1 within `tol`, or an invalid `base` (non-positive, `1.0`, NaN, or infinite).

```python
binary_entropy(p: float, base: float = 2.0) -> float
```
Shannon entropy of a Bernoulli(`p`). Returns `0.0` exactly at the boundaries `p=0`/`p=1`. Raises `ValueError` if `p` isn't in `[0, 1]` (NaN fails this range check naturally).

```python
joint_entropy(joint: Sequence[Sequence[Number]], base: float = 2.0) -> float
```
`H(X, Y)` of a 2D joint probability table — flattens the table and delegates to `entropy` (which also enforces the flattened table sums to 1).

### Joint-distribution quantities

```python
marginal_from_joint(joint, axis: int = 0) -> List[float]
```
Sums out one axis of a joint table. `axis=0` sums over rows (per-column/Y marginal); `axis=1` sums over columns (per-row/X marginal). Raises `ValueError` for `axis` not in `{0, 1}` or a malformed joint table.

```python
conditional_entropy(joint, given: str = "X", base: float = 2.0) -> float
```
`H(Y|X) = H(X,Y) - H(X)` (or the `X`/`Y` swap for `given="Y"`). Raises `ValueError` for `given` not in `{"X", "Y"}`.

```python
mutual_information(joint, base: float = 2.0) -> float
normalized_mutual_information(joint, base: float = 2.0) -> float
```
`I(X;Y) = H(X) + H(Y) - H(X,Y)`, clamped to `>= 0.0` to absorb floating-point noise. NMI divides by the **arithmetic mean** of `H(X)` and `H(Y)` (matching scikit-learn's `average_method='arithmetic'`) — one of several common normalization conventions (others use geometric mean, min, or max); returns `0.0` in the degenerate case where both marginals have zero entropy.

### Cross entropy / divergences

```python
cross_entropy(p, q, base: float = 2.0) -> float
```
`H(p, q) = -sum(p_i * log_base(q_i))`. Returns `float('inf')` if `q_i == 0` where `p_i > 0` (q assigns zero probability to an event p considers possible). Raises `ValueError` on length mismatch or either input not being a valid distribution.

```python
binary_cross_entropy(y_true, y_pred_proba, eps: float = 1e-15) -> float
```
Averaged binary log loss. Predicted probabilities are clipped to `[eps, 1-eps]` **only** to avoid `log(0)` at the exact boundary — values genuinely outside `[0, 1]` are rejected (`ValueError`) rather than silently clipped, since that usually indicates a bug in the caller's model rather than a legitimate boundary case. Also raises `ValueError` for length mismatch, non-binary `y_true` values, or NaN in `y_pred_proba`; returns `0.0` for empty inputs.

```python
kl_divergence(p, q, base: float = 2.0) -> float
```
`D_KL(p‖q) = sum(p_i * log_base(p_i/q_i))`. Returns `float('inf')` if `p_i > 0` where `q_i == 0`. Result is clamped to `>= 0.0` (Gibbs' inequality guarantees non-negativity; the clamp absorbs floating-point noise).

```python
js_divergence(p, q, base: float = 2.0) -> float
```
Symmetrized, smoothed KL divergence: `JSD(p,q) = 0.5*D_KL(p‖m) + 0.5*D_KL(q‖m)` where `m = (p+q)/2`. Always finite (unlike KL) and symmetric in `p`, `q`.

### Decision-tree splitting criteria

```python
information_gain(parent: Sequence[Number], subsets: Sequence[Sequence[Number]], base: float = 2.0) -> float
gini_gain(parent, subsets) -> float
```
Both share `_weighted_impurity_gain`: parent impurity minus the sample-weighted sum of child impurities, using `entropy` or `gini_impurity` respectively as the impurity function. A subset with zero total count is skipped (not an error). Raises `ValueError` for empty `parent`/`subsets`, negative/NaN counts, a subset whose category count doesn't match `parent`'s (a silent positional-alignment bug otherwise), or all-zero parent/subset totals.

```python
gini_impurity(p: Sequence[Number]) -> float
```
`1 - sum(p_i^2)` of a class distribution.

### Perplexity / generalized entropy

```python
perplexity(p, base: float = 2.0) -> float
```
`base^H(p)` — the effective number of equally-likely outcomes.

```python
renyi_entropy(p, alpha: float, base: float = 2.0) -> float
```
Generalizes Shannon entropy (`alpha -> 1`). Computed in log-space (log-sum-exp of `alpha * log(p_i)`) rather than via direct exponentiation of each `p_i**alpha`, specifically to avoid silent underflow to exactly `0.0` for large `alpha`. Special-cases `alpha` within `1e-10` of `1.0` (returns Shannon entropy directly) and `alpha = inf` (min-entropy limit `-log(max(p))`, handled separately rather than falling through to the log-space formula, since `alpha * log(p_i)` would be `inf * 0 = NaN` whenever any `p_i == 1.0` exactly). Raises `TypeError`/`ValueError` for non-numeric, boolean, non-positive, or NaN `alpha`.

---

## Example Session

```python
from information_theory import (
    entropy, binary_entropy, joint_entropy, mutual_information,
    kl_divergence, js_divergence, cross_entropy, binary_cross_entropy,
    information_gain, gini_impurity, gini_gain, perplexity, renyi_entropy,
)

entropy([0.5, 0.5])                          # 1.0 (bits)
binary_entropy(0.3)

joint = [[0.1, 0.2], [0.3, 0.4]]
joint_entropy(joint)
mutual_information(joint)

kl_divergence([0.5, 0.5], [0.9, 0.1])        # 0.7369...
js_divergence([0.1, 0.4, 0.5], [0.2, 0.3, 0.5])

binary_cross_entropy([1, 0, 1, 1, 0], [0.9, 0.1, 0.8, 0.6, 0.3])

# Decision-tree splitting
parent = [10, 10]
subsets = [[8, 2], [2, 8]]
information_gain(parent, subsets)
gini_gain(parent, subsets)
gini_impurity([0.5, 0.5])                     # 0.5

perplexity([0.2, 0.3, 0.5])
renyi_entropy([0.9, 0.05, 0.05], alpha=2.0)   # collision entropy
```

---

## Design Notes

- **NaN is explicitly checked everywhere**, not caught incidentally by
  range comparisons. `x < 0.0`, `x > 1.0`, and similar guards are all
  `False` when `x` is NaN in IEEE-754/Python, so a naive implementation
  would silently let NaN probabilities/counts/alphas flow into entropy
  calculations and produce a NaN result instead of a clear error. Every
  validation helper (`_validate_distribution`, `_validate_joint`,
  `_validate_counts`, `binary_entropy`, `renyi_entropy`) checks
  `math.isnan` directly, and this is locked in by dedicated regression
  tests (`test_nan_probability_raises`,
  `test_nan_parent_count_raises`, etc.).
- **`binary_cross_entropy` rejects out-of-range probabilities instead of
  silently clipping them.** An earlier version clipped predicted
  probabilities to `[0, 1]` unconditionally, which masks real bugs in a
  caller's model (a probability of `1.5` or `-0.1` is never legitimate).
  The function still clips to `[eps, 1-eps]` — but only to avoid
  `log(0)` exactly at the valid boundary, not to paper over invalid
  input; anything genuinely outside `[0, 1]` raises `ValueError`. This
  distinction is guarded by a regression test
  (`test_out_of_range_proba_raises`).
- **`information_gain`/`gini_gain` validate category alignment.** A
  subset with a different number of categories than the parent is a
  silent positional-alignment bug (comparing "class A count" in the
  parent against "class B count" in a child) rather than a legitimate
  edge case, so `_weighted_impurity_gain` raises `ValueError` with an
  explicit message rather than producing a numerically-plausible but
  meaningless result.
- **`renyi_entropy` computes in log-space to avoid underflow.** For
  large `alpha`, direct exponentiation of `p_i**alpha` underflows to
  exactly `0.0` for every `i` (since `p_i < 1`), which would make the
  entropy calculation incorrectly return `0.0` instead of converging
  toward the true min-entropy limit `-log(max(p))`. Using log-sum-exp on
  `alpha * log(p_i)` avoids this entirely; a dedicated regression test
  (`test_large_alpha_approaches_min_entropy_without_underflow`) checks
  `alpha=1000` still gives the correct answer and isn't `0.0`.
- **`alpha = inf` is a genuine special case, not a fallthrough.** The
  general log-space formula involves `alpha * log(p_i)`, which becomes
  `inf * 0 = NaN` whenever any `p_i` is exactly `1.0` (a legitimate,
  common input — e.g. a deterministic distribution). The min-entropy
  limit is therefore computed directly (`-log(max(p))`) rather than
  relying on the general formula to handle the limit correctly.
- **KL and cross-entropy return `inf`, not raise, for the classic
  zero-support mismatch.** When `q_i == 0` but `p_i > 0`, KL divergence
  and cross entropy are mathematically infinite (`q` assigns zero
  probability to an event `p` considers possible) — this is a valid,
  meaningful answer, not an error condition, so both functions return
  `float('inf')` rather than raising.
- **Jensen-Shannon divergence exists specifically because KL isn't
  always usable.** JSD is built from two KL divergences against the
  midpoint distribution `m = (p+q)/2`, which is guaranteed to have full
  support wherever `p` or `q` does — this keeps JSD finite and symmetric
  even for `p`/`q` with completely disjoint support, unlike raw KL
  divergence (tested directly via
  `test_js_divergence_finite_even_when_disjoint_support`).
- **Log dispatch avoids the generic two-argument `math.log(x, base)`**
  for the common bases (2, e, 10), instead calling `math.log2`/
  `math.log`/`math.log10` directly — each is slightly more accurate for
  its respective base (fewer floating-point rounding steps), which
  compounds meaningfully when an entropy calculation sums many log
  evaluations.
- **All floating-point-noise clamps are one-directional and
  well-justified**, not blanket clipping: `kl_divergence`'s result is
  clamped to `>= 0.0` because Gibbs' inequality guarantees
  non-negativity mathematically; `mutual_information`/
  `conditional_entropy`/`_weighted_impurity_gain` clamp similarly for
  quantities that are provably non-negative. These clamps only correct
  for accumulated floating-point rounding, never for a genuine
  out-of-range input (which is caught by validation instead).

---

## Test Coverage

`tests/test_information_theory.py` cross-checks numeric correctness
against `scipy.stats.entropy` (Shannon entropy and KL divergence) and
`scipy.spatial.distance.jensenshannon` (JS divergence, noting SciPy
returns the square root of JSD) in addition to covering: validation
(non-numeric/boolean/NaN/Inf rejection, sum-to-1 tolerance, invalid
bases), joint-distribution consistency (mutual information/conditional
entropy derived manually from marginals and compared against the direct
functions), independence and perfect-dependence edge cases for mutual
information/NMI, decision-tree splitting correctness (perfect vs.
useless splits, category-count mismatch, NaN-count rejection), and the
Renyi-entropy underflow regression at large `alpha`.

Run with verbose output:

```bash
pytest tests/test_information_theory.py -v
```

---

## Roadmap Context

`InformationTheory/` is a self-contained module in **Phase 1: Math
Foundations** — unlike `Matrix`, `Probability`, `Bayesian`, and
`Optimization`, it has no dependencies on the other math-foundations
modules, reflecting that entropy/divergence/splitting-criteria math is
usable in isolation (e.g. directly inside a from-scratch decision tree
or classifier implementation). It carries forward the same conventions
established elsewhere in the curriculum — explicit NaN/boolean
rejection, one-directional floating-point-noise clamps with documented
justification, full docstring + type-hint coverage — and is expected to
be consumed directly by later from-scratch ML algorithm implementations
(decision trees, random forests, classifiers using cross-entropy loss).
