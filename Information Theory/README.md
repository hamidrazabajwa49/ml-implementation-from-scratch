# Information Theory Module

Built from scratch in pure Python — no NumPy, no SciPy.
Part of a 60-project ML/AI foundations series.

---

## Overview

Core information theory primitives used throughout machine learning: entropy measures, divergences, and decision tree splitting criteria. Every function is tested against mathematical identities (symmetry, non-negativity, boundary conditions).

---

## Functions

### Entropy

**`entropy(p, base=2.0)`**
Shannon entropy of a discrete distribution. Measures average uncertainty.
```
H(X) = -Σ p(x) log p(x)
```
```python
entropy([0.5, 0.5])          # 1.0 bit  — maximum uncertainty
entropy([1.0, 0.0])          # 0.0 bits — certain outcome
entropy([0.25, 0.25, 0.25, 0.25])  # 2.0 bits
```

**`binary_entropy(p, base=2.0)`**
Shorthand for a Bernoulli variable. `H(p) = -p log p - (1-p) log(1-p)`.
```python
binary_entropy(0.5)   # 1.0  — coin flip
binary_entropy(0.1)   # 0.469
```

**`renyi_entropy(p, alpha, base=2.0)`**
Generalization of Shannon entropy. `alpha=1` recovers Shannon exactly.
```python
renyi_entropy([0.5, 0.3, 0.2], alpha=2.0)  # collision entropy
renyi_entropy([0.5, 0.3, 0.2], alpha=1.0)  # == entropy(...)
```

---

### Joint & Conditional

**`joint_entropy(joint, base=2.0)`**
Entropy of a joint distribution `P(X, Y)` passed as a 2D list.
```python
# Independent: H(X,Y) = H(X) + H(Y)
joint_entropy([[0.25, 0.25], [0.25, 0.25]])  # 2.0

# Fully correlated: H(X,Y) = H(X)
joint_entropy([[0.5, 0.0], [0.0, 0.5]])      # 1.0
```

**`conditional_entropy(joint, given='X', base=2.0)`**
`H(Y|X)` or `H(X|Y)`. Uses chain rule: `H(Y|X) = H(X,Y) - H(X)`.
```python
conditional_entropy(joint, given='X')  # H(Y|X)
conditional_entropy(joint, given='Y')  # H(X|Y)
```

**`mutual_information(joint, base=2.0)`**
Shared information between X and Y. `I(X;Y) = H(X) + H(Y) - H(X,Y)`.
```python
mutual_information([[0.25,0.25],[0.25,0.25]])  # 0.0 — independent
mutual_information([[0.5,0.0],[0.0,0.5]])      # 1.0 — fully correlated
```

**`normalized_mutual_information(joint, base=2.0)`**
MI scaled to `[0, 1]`. Used to evaluate clustering quality.
```
NMI = I(X;Y) / (0.5 * (H(X) + H(Y)))
```

---

### Divergences

**`cross_entropy(p, q, base=2.0)`**
Expected log-loss of predicting `q` when truth is `p`. The loss function in classification.
```
H(p, q) = -Σ p(x) log q(x)
```
```python
cross_entropy([0.7, 0.3], [0.9, 0.1])   # low — q close to p
cross_entropy([0.7, 0.3], [0.1, 0.9])   # high — q far from p
```
Returns `inf` if `q(x) = 0` where `p(x) > 0`.

**`kl_divergence(p, q, base=2.0)`**
How much `q` diverges from `p`. Asymmetric. Always ≥ 0.
```
KL(p||q) = Σ p(x) log(p(x)/q(x))
         = H(p, q) - H(p)
```
```python
kl_divergence([0.7, 0.3], [0.4, 0.6])   # KL(p||q)
kl_divergence([0.4, 0.6], [0.7, 0.3])   # different — asymmetric
```

**`js_divergence(p, q, base=2.0)`**
Symmetric, bounded version of KL divergence. `JSD ∈ [0, 1]` (base 2).
```
JSD(p||q) = 0.5 KL(p||m) + 0.5 KL(q||m),   m = (p+q)/2
```
```python
js_divergence(p, q) == js_divergence(q, p)   # symmetric ✓
js_divergence([1,0], [0,1])                  # 1.0 — maximum
js_divergence(p, p)                          # 0.0
```

---

### Decision Tree Criteria

**`information_gain(parent, subsets, base=2.0)`**
Entropy reduction after splitting. Primary criterion for ID3 / C4.5.
```
IG = H(parent) - Σ (|subset| / |total|) × H(subset)
```
```python
parent  = [50, 50]              # 50 class-A, 50 class-B
left    = [45, 5]               # mostly class-A
right   = [5,  45]              # mostly class-B
information_gain(parent, [left, right])   # high — good split
information_gain(parent, [parent, parent])  # 0.0 — no improvement
```

**`gini_impurity(p)`**
Probability of misclassifying a random sample. Used in CART.
```
Gini(p) = 1 - Σ p(x)²
```
```python
gini_impurity([1.0, 0.0])       # 0.0   — pure node
gini_impurity([0.5, 0.5])       # 0.5   — maximum impurity
gini_impurity([0.25]*4)         # 0.75
```

**`gini_gain(parent, subsets)`**
Gini impurity reduction after a split.
```python
gini_gain([50, 50], [[45, 5], [5, 45]])   # high — good split
```

**`perplexity(p, base=2.0)`**
Effective vocabulary size of a distribution. Standard metric for language models.
```
PP(p) = base^{H(p)}
```
```python
perplexity([1/8]*8)    # 8.0  — uniform over 8 outcomes
perplexity([1.0, 0.0]) # 1.0  — no uncertainty
```

---

## Mathematical Identities (verified by tests)

```
H(X,Y)   = H(X) + H(Y|X)            chain rule
I(X;Y)   = H(X) + H(Y) - H(X,Y)     mutual information
KL(p||q) = H(p,q) - H(p)            KL via cross-entropy
JSD(p,q) = JSD(q,p)                  symmetry
JSD      ∈ [0, 1]                    bounded (base 2)
IG       ≥ 0                          non-negative
Gini     ∈ [0, 1 - 1/k]              bounded
PP(p)    = base^{H(p)}               perplexity definition
Rényi(α=1) = Shannon                  limit identity
```

---

## Used in Later Projects

| Project | Function |
|---|---|
| Decision Trees | `information_gain`, `gini_impurity`, `gini_gain` |
| Naive Bayes | `entropy`, `cross_entropy` |
| Neural Networks | `cross_entropy` (loss), `kl_divergence` |
| Clustering eval | `normalized_mutual_information` |
| Language Models | `perplexity`, `cross_entropy` |
| VAE / GANs | `kl_divergence`, `js_divergence` |

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

→ [GitHub](https://github.com/hamidrazabajwa49/ml-implementation-from-scratch) | [LinkedIn](www.linkedin.com/in/hamid-raza-bajwa-564a91377)
