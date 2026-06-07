# Probability — Pure Python Probability & Statistics Library

## Overview

The `Probability/` package is a pure Python implementation of probability distributions, descriptive statistics, special functions, and hypothesis tests — built from scratch as part of the *Engineering Redemption Arc*, a structured 60-project ML engineering curriculum in the [`ml-implementation-from-scratch`](https://github.com/) repository.

The package covers the statistical foundation required before implementing ML algorithms: summary statistics, continuous and discrete distributions, numerical special functions, and inferential tests. All computations use Python's standard library only — no NumPy, SciPy, or Pandas.

The `Matrix` class from the `Matrix/` package is used for covariance and correlation matrix construction.

---

## Project Structure

```
ml-implementation-from-scratch/
├── Matrix/
│   └── matrix.py                  # Matrix primitive (required dependency)
├── Vectors/
│   └── vector.py                  # Vector primitive (transitive dependency)
└── Probability/
    ├── distributions.py           # Abstract base class for all distributions
    ├── discrete.py                # Bernoulli, Binomial, Poisson
    ├── continuous.py              # Normal, T, Chi-squared, F, Beta, Gamma
    ├── special_functions.py       # Regularized incomplete Beta and Gamma functions
    ├── descriptive_stats.py       # DescriptiveStats class + covariance/correlation matrices
    └── hypothesis_tests.py        # t-tests, chi-square tests, one-way ANOVA
```

Each file's internal path resolution (`sys.path.insert`) assumes this layout. Moving files out of the `Probability/` directory will break imports.

---

## Dependencies

| Requirement | Detail |
|---|---|
| Python | 3.10+ |
| `math`, `random`, `os`, `sys`, `cmath`, `collections` | Standard library only |
| `Matrix/matrix.py` | Local dependency — covariance/correlation matrices |
| `Vectors/vector.py` | Transitive dependency via `matrix.py` |

No `pip install` required.

---

## Installation

Ensure the folder layout above is intact. Run scripts from the repository root, or insert the root into `sys.path`:

```python
import sys
sys.path.insert(0, "/path/to/ml-implementation-from-scratch")

from Probability.descriptive_stats import DescriptiveStats
from Probability.continuous import NormalDistribution
from Probability.discrete import BinomialDistribution
from Probability.hypothesis_tests import ttest_1samp
from Probability.special_functions import betainc, gammainc
```

---

## Module Reference

---

### `distributions.py` — Abstract Base Class

`Distribution` is the base class for all discrete and continuous distribution classes. It defines a shared interface:

```python
dist.mean()       # float
dist.variance()   # float
dist.std()        # float — default: sqrt(variance())
dist.sample(n)    # list of n samples
dist.summary()    # formatted string: name, mean, variance, std
```

Subclasses must implement `mean()`, `variance()`, and `sample()`. All other methods have default implementations.

---

### `discrete.py` — Discrete Distributions

All discrete distributions inherit from `Distribution`.

#### `BinomialDistribution(n, p)`

Models the number of successes in `n` independent Bernoulli trials with success probability `p`.

```python
dist = BinomialDistribution(n=10, p=0.3)
dist.pmf(k=3)        # P(X = 3)
dist.cdf(x=4)        # P(X <= 4)
dist.mean()          # n * p
dist.variance()      # n * p * (1 - p)
dist.sample(100)     # list of 100 samples
```

`k` must be an integer. Returns `0.0` for `k` outside `[0, n]`.

#### `BernoulliDistribution(p)`

Special case of Binomial with `n=1`. Outcomes are `0` or `1`.

```python
dist = BernoulliDistribution(p=0.6)
dist.pmf(1)      # 0.6
dist.pmf(0)      # 0.4
dist.cdf(0.5)    # 0.4
dist.sample(10)  # list of 0s and 1s
```

#### `PoissonDistribution(lam)`

Models the number of events in a fixed interval given average rate `lam`.

```python
dist = PoissonDistribution(lam=4.5)
dist.pmf(k=3)    # P(X = 3)
dist.cdf(x=5)    # P(X <= 5)
dist.mean()      # lam
dist.variance()  # lam
dist.sample(50)
```

`k` must be an integer. `lam=0` is handled as a degenerate distribution at zero.

---

### `continuous.py` — Continuous Distributions

Continuous distributions do **not** inherit from `Distribution`; they implement their own `pdf`, `cdf`, `sf`, `ppf`, `mean`, `variance`, and `std` methods directly.

#### `NormalDistribution(mu, sigma)`

```python
dist = NormalDistribution(mu=0.0, sigma=1.0)
dist.pdf(x)      # probability density
dist.cdf(x)      # P(X <= x) via math.erf
dist.sf(x)       # 1 - cdf(x)
dist.ppf(p)      # inverse CDF (quantile) — Rational approximation (Peter Acklam's method)
dist.mean()      # mu
dist.variance()  # sigma^2
dist.std()       # sigma
```

`sigma` must be positive. `ppf` requires `p` strictly in `(0, 1)`.

#### `TDistribution(df)`

```python
dist = TDistribution(df=10)
dist.pdf(t)
dist.cdf(t)          # via regularized incomplete Beta function
dist.sf(t)
dist.ppf(p)          # via bisection search
dist.p_value(t, alternative)   # 'two-sided', 'greater', or 'less'
```

#### `Chi2Distribution(df)`

```python
dist = Chi2Distribution(df=5)
dist.pdf(x)
dist.cdf(x)    # via regularized incomplete Gamma function
dist.sf(x)
```

Returns `0.0` for `x <= 0`.

#### `FDistribution(df1, df2)`

```python
dist = FDistribution(df1=3, df2=20)
dist.pdf(f)
dist.cdf(f)    # via regularized incomplete Beta function
dist.sf(f)
```

Returns `0.0` for `f <= 0`.

#### `BetaDistribution(alpha, beta)`

```python
dist = BetaDistribution(alpha=2.0, beta=5.0)
dist.pdf(x)          # log-space computation; handles boundary singularities
dist.cdf(x)          # trapezoidal integration (2000 steps)
dist.ppf(p)          # bisection over [0, 1]
dist.mean()          # alpha / (alpha + beta)
dist.variance()
dist.std()
dist.sample(n)       # Johnk's method — valid for all alpha, beta > 0
```

#### `GammaDistribution(alpha, beta)`

Rate parameterization: mean = `alpha / beta`.

```python
dist = GammaDistribution(alpha=3.0, beta=1.0)
dist.pdf(x)
dist.cdf(x)          # trapezoidal integration
dist.ppf(p)          # bisection with adaptive upper bound
dist.mean()          # alpha / beta
dist.variance()      # alpha / beta^2
dist.std()
dist.sample(n)       # Marsaglia-Tsang method; handles alpha < 1 via boosting
```

---

### `special_functions.py` — Numerical Special Functions

Low-level numerical routines used internally by `TDistribution`, `Chi2Distribution`, and `FDistribution`. Can also be called directly.

#### `betainc(a, b, x)`

Regularized incomplete Beta function I_x(a, b) — returns P(X ≤ x) for X ~ Beta(a, b).

```python
from Probability.special_functions import betainc
betainc(2.0, 5.0, 0.3)   # float in [0, 1]
```

Implemented via Lentz's continued fraction expansion with symmetry reflection for numerical stability. Converges to `1e-15` tolerance within 200 iterations.

#### `gammainc(a, x)`

Regularized lower incomplete Gamma function P(a, x) — returns P(X ≤ x) for X ~ Gamma(a, 1).

```python
from Probability.special_functions import gammainc
gammainc(3.0, 2.0)    # float in [0, 1]
```

Selects between series expansion (for `x < a + 1`) and continued fraction (`x >= a + 1`) for numerical stability. Converges to `1e-12` tolerance within 300 iterations.

---

### `descriptive_stats.py` — Descriptive Statistics

#### `DescriptiveStats(data)`

Accepts a non-empty list of `int` or `float` values. Data is sorted internally; the original order is preserved in `._raw`.

```python
from Probability.descriptive_stats import DescriptiveStats

ds = DescriptiveStats([4, 7, 2, 9, 1, 5])

ds.mean()              # arithmetic mean
ds.median()            # 50th percentile
ds.mode()              # list of most frequent values (sorted)
ds.percentile(p)       # linear interpolation; p in [0, 100]
ds.variance(ddof=0)    # population variance (ddof=0) or sample variance (ddof=1)
ds.std(ddof=0)         # standard deviation
ds.data_range()        # max - min
ds.iqr()               # Q3 - Q1
ds.skewness()          # Fisher's moment coefficient
ds.kurtosis()          # excess kurtosis (normal = 0)
ds.summary()           # formatted string of all statistics
```

#### Static Methods

```python
DescriptiveStats.covariance(x, y, ddof=0)       # float
DescriptiveStats.correlation(x, y)              # Pearson r; float in [-1, 1]
DescriptiveStats.covariance_matrix(dataset)     # Matrix object (ddof=1)
DescriptiveStats.correlation_matrix(dataset)    # Matrix object
```

`dataset` is a list of lists — one list per variable. All variables must have the same length. Returns a `Matrix` instance from the `Matrices/` package.

```python
x = [1, 2, 3, 4]
y = [2, 4, 5, 4]
z = [1, 3, 5, 2]

cov_mat = DescriptiveStats.covariance_matrix([x, y, z])   # 3×3 Matrix
cor_mat = DescriptiveStats.correlation_matrix([x, y, z])  # 3×3 Matrix
```

---

### `hypothesis_tests.py` — Inferential Tests

All test functions return a result `dict` with consistent keys: `t_statistic` or `chi2_statistic` or `F_statistic`, `p_value`, `df`, `alpha`, `reject_H0`, and `conclusion`.

#### One-Sample t-test

```python
from Probability.hypothesis_tests import ttest_1samp

result = ttest_1samp(
    data=[2.1, 2.5, 2.3, 2.8, 2.0],
    pop_mean=2.0,
    alternative="two-sided",   # 'two-sided', 'greater', or 'less'
    alpha=0.05
)
# result keys: t_statistic, p_value, df, alpha, reject_H0, alternative, conclusion
```

Detects near-constant data and raises `ValueError` before computing an undefined t-statistic.

#### Independent Samples t-test (Welch's)

```python
from Probability.hypothesis_tests import ttest_ind

result = ttest_ind(data1, data2, alternative="two-sided", alpha=0.05)
# Uses Welch-Satterthwaite df approximation
```

#### Paired t-test

```python
from Probability.hypothesis_tests import ttest_paired

result = ttest_paired(before, after, alternative="less", alpha=0.05)
# Internally: ttest_1samp on element-wise differences
```

#### Confidence Interval

```python
from Probability.hypothesis_tests import confidence_interval

result = confidence_interval(data, confidence=0.95)
# result keys: mean, std, n, confidence, t_critical, margin, lower, upper
```

#### Chi-Square Goodness of Fit

```python
from Probability.hypothesis_tests import chisquare_gof

result = chisquare_gof(
    observed=[18, 22, 20, 15, 25],
    expected=None,    # None = uniform expected frequencies
    alpha=0.05
)
```

#### Chi-Square Test of Independence

```python
from Probability.hypothesis_tests import chisquare_independence

table = [[10, 20, 30],
         [6,  9,  17]]

result = chisquare_independence(table, alpha=0.05)
# result also includes: expected_table
```

#### One-Way ANOVA

```python
from Probability.hypothesis_tests import anova_oneway

result = anova_oneway(group_a, group_b, group_c, alpha=0.05)
# result keys: F_statistic, p_value, df_between, df_within,
#              SSB, SSW, MSB, MSW, group_means, grand_mean,
#              alpha, reject_H0, conclusion
```

Requires at least two groups, each with at least two observations. Handles zero within-group variance edge cases.

---

## Example Session

```python
from Probability.descriptive_stats import DescriptiveStats
from Probability.continuous import NormalDistribution, TDistribution
from Probability.discrete import PoissonDistribution
from Probability.hypothesis_tests import ttest_ind, anova_oneway, confidence_interval

data = [14.2, 13.8, 15.1, 14.7, 13.5, 15.3, 14.9]
ds = DescriptiveStats(data)
print(ds.summary())

norm = NormalDistribution(mu=14.5, sigma=0.6)
print(norm.cdf(15.0))      # P(X <= 15)
print(norm.ppf(0.975))     # 97.5th percentile

pois = PoissonDistribution(lam=3.0)
print(pois.pmf(k=2))
print(pois.sample(10))

result = ttest_ind(
    [14.2, 13.8, 15.1, 14.7],
    [13.1, 12.9, 13.5, 14.0],
    alternative="two-sided"
)
print(result["conclusion"])

ci = confidence_interval(data, confidence=0.95)
print(f"95% CI: ({ci['lower']:.3f}, {ci['upper']:.3f})")

result = anova_oneway([5, 6, 7], [8, 9, 10], [4, 5, 6], alpha=0.05)
print(result["F_statistic"], result["p_value"])
```

---

## Error Reference

| Situation | Exception |
|---|---|
| `sigma <= 0` in NormalDistribution | `ValueError` |
| `df <= 0` in T/Chi2/F distributions | `ValueError` |
| `alpha <= 0` or `beta <= 0` in Beta/Gamma | `ValueError` |
| `p` outside `(0, 1)` in `ppf` | `ValueError` |
| `alternative` not in allowed set | `ValueError` |
| `alpha` outside `(0, 1)` in hypothesis tests | `ValueError` |
| Non-integer `k` in `pmf` (Binomial, Poisson) | `TypeError` |
| Empty or non-numeric data in `DescriptiveStats` | `ValueError` / `TypeError` |
| `ddof=1` with fewer than 2 points | `ValueError` |
| Mismatched list lengths in `covariance`/`correlation` | `ValueError` |
| Near-constant data in `ttest_1samp` | `ValueError` |
| Paired samples of unequal length | `ValueError` |
| Zero grand total in chi-square tests | `ValueError` |
| Zero expected count in independence test | `ValueError` |
| Fewer than 2 groups or observations in ANOVA | `ValueError` |
| `x` outside `[0, 1]` in `betainc` | `ValueError` |
| `x < 0` in `gammainc` | `ValueError` |

---

## Design Notes

- **No external libraries:** All numerical methods — continued fraction expansion for the incomplete Beta, series/CF for the incomplete Gamma, Rational approximation for the Normal PPF, Marsaglia-Tsang sampling for Gamma — are implemented from scratch.
- **Log-space arithmetic:** PDF computations for Beta, Gamma, T, F, and Chi-squared distributions use `math.lgamma` and `math.log` to avoid overflow and underflow on extreme parameter values.
- **Numerical stability in special functions:** `betainc` uses the symmetry relation `I_x(a,b) = 1 - I_{1-x}(b,a)` to route evaluations to the faster-converging side of the continued fraction. `gammainc` switches between series and CF based on the `x < a+1` heuristic.
- **`DescriptiveStats` sorts on construction:** All percentile and IQR computations operate on a sorted copy; the unsorted original is preserved in `._raw`.
- **Welch's t-test, not Student's:** `ttest_ind` uses the Welch-Satterthwaite degrees of freedom approximation — no assumption of equal variances.
- **`BetaDistribution.cdf` and `GammaDistribution.cdf` use trapezoidal integration:** These are numerically approximate (2000 steps). For high-precision needs, use `betainc`/`gammainc` directly.

---

## Roadmap Context

This package spans multiple foundational projects in the Engineering Redemption Arc curriculum:

- **Depends on** — `Vectors/vector.py` (Project 1), `Matrices/matrix.py` (Project 2)
- **Underpins** — Bayesian inference, information theory, and all ML model implementations in Phase 2 (Projects 11–25), where probability distributions, log-likelihoods, and hypothesis tests are used directly in model training and evaluation.
