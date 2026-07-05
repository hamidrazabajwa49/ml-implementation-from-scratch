# Probability — Distributions, Descriptive Stats & Hypothesis Testing (From Scratch)

**Phase 1: Math Foundations**.

A dependency-free, pure-Python probability and statistics library:
descriptive statistics (including covariance/correlation matrices via the
`Matrix` module), discrete and continuous probability distributions, the
special functions that back their exact CDFs, and classical frequentist
hypothesis tests built on top of all three. SciPy/NumPy are used only in
the test suite, as correctness oracles.

---

## Overview

The package is organized into five files, each building on the ones
before it:

| File | Provides |
|---|---|
| `special_functions.py` | Regularized incomplete beta (`betainc`) and gamma (`gammainc`) functions — the numerical backbone for exact CDFs |
| `distributions.py` | `Distribution` — the abstract base class shared by every distribution |
| `discrete.py` | `BernoulliDistribution`, `BinomialDistribution`, `PoissonDistribution` |
| `continuous.py` | `NormalDistribution`, `TDistribution`, `Chi2Distribution`, `FDistribution`, `BetaDistribution`, `GammaDistribution` |
| `descriptive_stats.py` | `DescriptiveStats` — single-dataset summaries plus covariance/correlation (matrices via `Matrix`) |
| `hypothesis_tests.py` | Free functions: t-tests, z-test, confidence intervals, chi-squared tests, one-way ANOVA |

Every distribution exposes the same interface (`mean`, `variance`, `std`,
`sample`, `summary`, plus `pmf`/`pdf`, `cdf`, `sf`, `ppf` where
mathematically defined), so downstream code — including
`hypothesis_tests.py` — can treat any distribution polymorphically.

---

## Project Structure

```
math_foundations/
└── Probability/
    ├── special_functions.py
    ├── distributions.py
    ├── discrete.py
    ├── continuous.py
    ├── descriptive_stats.py
    ├── hypothesis_tests.py
    ├── tests/
    │   ├── test_special_functions.py
    │   ├── test_discrete.py
    │   ├── test_continuous.py
    │   ├── test_descriptive_stats.py
    │   └── test_hypothesis_tests.py
    └── README.md
```

`descriptive_stats.py` imports `Matrix` from the sibling `Matrix` module
(`math_foundations/Matrix/matrix.py`); `discrete.py`/`continuous.py`
import `Distribution` from `distributions.py` and `betainc`/`gammainc`
from `special_functions.py`; `hypothesis_tests.py` imports from
`descriptive_stats.py` and `continuous.py` — all via relative `sys.path`
inserts. `Vectors`, `Matrix`, and `Probability` must live side-by-side
under `math_foundations/`.

---

## Dependencies

| Component | Requires |
|---|---|
| Library files | Python 3.8+ standard library only (`math`, `random`, `warnings`, `collections`, `abc`, `typing`) + `Matrix.matrix.Matrix` (sibling module) |
| `tests/*.py` | `pytest`, `numpy`, `scipy` (test-only, for regression checks) |

Install test dependencies:

```bash
pip install pytest numpy scipy
```

Run the full suite from `math_foundations/Probability/`:

```bash
pytest tests/ -v
```

Or an individual file:

```bash
pytest tests/test_continuous.py -v
```

---

## Module Reference

### `special_functions.py`

```python
betainc(a: float, b: float, x: float, max_iter=200, tol=1e-15) -> float
```
Regularized incomplete beta function `I_x(a, b)`, via Lentz's modified continued fraction (Numerical Recipes §6.4), using the symmetry relation `I_x(a,b) = 1 - I_{1-x}(b,a)` to guarantee fast convergence regardless of which side of `x < (a+1)/(a+b+2)` the input falls on.

```python
betaincc(a, b, x, max_iter=200, tol=1e-15) -> float
```
Complement `1 - I_x(a, b)`, computed directly (via the symmetry relation) rather than by subtracting `betainc` from 1, for better accuracy near 0/1.

```python
gammainc(a: float, x: float, max_iter=300, tol=1e-12) -> float
gammaincc(a, x, max_iter=300, tol=1e-12) -> float
```
Regularized lower/upper incomplete gamma functions `P(a, x)` / `Q(a, x)`, via Numerical Recipes §6.2: a power series for `x < a + 1`, a continued fraction for `x >= a + 1`.

All four raise `TypeError` for non-numeric or boolean inputs, and `ValueError` for non-positive/non-finite/NaN shape parameters or an `x` outside its valid range (`[0, 1]` for beta, `>= 0` for gamma). Non-convergence logs a warning rather than raising, since the result is still the best available approximation.

### `distributions.py`

```python
class Distribution(ABC):
    def mean(self) -> float                                    # abstract
    def variance(self) -> float                                # abstract
    def std(self) -> float                                      # sqrt(variance())
    def sample(self, num_samples=1, seed=None) -> List[float]  # abstract
    def summary(self) -> str                                    # "ClassName: mean=..., variance=..., std=..."
```
The shared base class. `std()` is deliberately not meant to be overridden independently of `variance()`, to avoid mean/variance/std inconsistency bugs.

### `discrete.py`

| Class | Constructor | Notes |
|---|---|---|
| `BernoulliDistribution(p)` | `p` in `[0, 1]` | `pmf`, `cdf`, `sf`, `ppf`, `sample` all closed-form |
| `BinomialDistribution(n, p)` | `n` non-negative int, `p` in `[0, 1]` | `cdf`/`ppf` via direct summation, O(k); `sample` simulates `n` Bernoulli trials per draw |
| `PoissonDistribution(lam)` | `lam >= 0`, finite | `pmf` computed in log-space to avoid overflow; `sample` uses Knuth's algorithm, decomposed into safe-sized chunks above `lam=20` to avoid the `exp(-lam)` underflow that would otherwise hang the loop |

All three reject `bool` and non-finite/NaN parameters explicitly.

### `continuous.py`

| Class | Constructor | Notes |
|---|---|---|
| `NormalDistribution(mu=0, sigma=1)` | `sigma > 0` | `cdf`/`sf` via `erf`/`erfc`; `ppf` via the Acklam algorithm + one Halley refinement step (~machine precision) |
| `TDistribution(df)` | `df > 0` | `cdf` via `betainc`; `ppf` via bracketing + bisection (no closed form); `mean`/`variance` raise `ValueError` where undefined (`df <= 1` / `df <= 1`); also exposes `p_value(t, alternative)` |
| `Chi2Distribution(df)` | `df > 0` | `cdf` via `gammainc`; `ppf` via bracketing + bisection; `sample` via the Chi2 = Gamma(df/2, rate=1/2) relationship |
| `FDistribution(df1, df2)` | both `> 0` | `cdf` via `betainc`; `mean`/`variance` raise `ValueError` where undefined (`df2 <= 2` / `df2 <= 4`); `sample` via the ratio-of-scaled-Chi2s construction |
| `BetaDistribution(alpha, beta)` | both `> 0` | `cdf` **exact** via `betainc` (not trapezoidal integration); `sample` via Jöhnk's rejection method (acceptance degrades for large shape parameters — a documented limitation) |
| `GammaDistribution(alpha, beta)` | both `> 0`, shape-rate parameterization | `cdf` **exact** via `gammainc`; `sample` via Marsaglia-Tsang, boosted for `alpha < 1` |

`Chi2`/`Beta`/`Gamma` CDFs were specifically rewritten to use the exact incomplete beta/gamma functions instead of numerical (trapezoidal) integration — both far faster and more accurate; regression tests (`test_cdf_exact_matches_scipy`, `test_cdf_speed_is_fast`) guard against regressing to the old approach.

### `descriptive_stats.py`

```python
class DescriptiveStats:
    def __init__(self, data: Sequence[Number])
```
Validates non-empty, numeric, non-boolean, non-NaN input; stores a sorted copy (`self.data`) alongside the original order (`self._raw`).

**Central tendency:** `mean()`, `median()` (= `percentile(50)`), `mode()` (all values tied for highest frequency), `percentile(p)` (linear interpolation, matches `numpy.percentile`'s default method).

**Spread:** `variance(ddof=0|1)`, `std(ddof=0|1)`, `data_range()`, `iqr()`.

**Shape:** `skewness()`, `kurtosis()` (population/biased Fisher-Pearson; both return `0.0` for constant data by convention rather than raising).

**Standardization:** `z_scores(ddof=0)` — returned in the *original* input order, not sorted.

**Bivariate/multivariate (static methods):**
```python
DescriptiveStats.covariance(x, y, ddof=0) -> float
DescriptiveStats.correlation(x, y) -> float                       # 0.0 if either series is constant
DescriptiveStats.covariance_matrix(dataset, ddof=1) -> Matrix      # symmetric p x p
DescriptiveStats.correlation_matrix(dataset) -> Matrix             # symmetric p x p, diagonal exactly 1.0
```
The matrix variants exploit symmetry (compute each off-diagonal pair once) and return a `Matrix` instance from the `Matrix` module.

**Reporting:** `summary()` — formatted multi-line text block with count, mean, median, std, variance, min, max, range, IQR, skewness, kurtosis.

### `hypothesis_tests.py`

Every function returns a plain `dict` (not a custom result class) with the statistic, p-value, and supporting details.

```python
ttest_1samp(data, pop_mean, alternative="two-sided", alpha=0.05) -> dict
```
One-sample t-test. Keys: `t_statistic`, `p_value`, `df`, `alpha`, `reject_H0`, `alternative`, `cohens_d`, `conclusion`. Raises `ValueError` for fewer than 2 observations, zero/negligible sample variance, or invalid `alternative`/`alpha`.

```python
ttest_ind(data1, data2, alternative="two-sided", alpha=0.05) -> dict
```
Welch's two-sample t-test (unequal variances assumed). `df` is the fractional Welch-Satterthwaite degrees of freedom. `cohens_d` uses the conventional pooled-standard-deviation effect size even though the test statistic itself is Welch's.

```python
ttest_paired(data1, data2, alternative="two-sided", alpha=0.05) -> dict
```
One-sample t-test on the within-pair differences. Raises `ValueError` on length mismatch or empty input.

```python
ztest_1samp(data, pop_mean, pop_std, alternative="two-sided", alpha=0.05) -> dict
```
One-sample z-test for known population standard deviation. Raises `ValueError` if `pop_std <= 0`.

```python
confidence_interval(data, confidence=0.95) -> dict
```
t-based confidence interval for the population mean. Keys: `mean`, `std`, `n`, `confidence`, `t_critical`, `margin`, `lower`, `upper`.

```python
chisquare_gof(observed, expected=None, alpha=0.05) -> dict
```
Chi-squared goodness-of-fit test. `expected` defaults to a uniform split of the observed total. Emits a `UserWarning` and returns `p_value=1.0` for a single category (zero degrees of freedom — the test cannot detect any departure).

```python
chisquare_independence(observed, alpha=0.05) -> dict
```
Chi-squared test of independence on a 2D contingency table. Returns `expected_table` alongside the usual statistic/p-value/df. Raises `TypeError` if rows aren't sequences, `ValueError` for ragged rows, negative counts, a zero grand total, or a row/column that sums to zero.

```python
anova_oneway(*groups, alpha=0.05) -> dict
```
One-way ANOVA F-test across 2+ independent groups (each needing >= 2 observations). Keys include `F_statistic`, `p_value`, `df_between`, `df_within`, `SSB`, `SSW`, `MSB`, `MSW`, `group_means`, `grand_mean`. Handles the degenerate within-group-variance-zero case explicitly: `F=NaN, p=NaN` if between-group variance is also zero (identical groups), `F=inf, p=0.0` otherwise (perfect separation).

---

## Example Session

```python
from descriptive_stats import DescriptiveStats
from discrete import BinomialDistribution
from continuous import NormalDistribution, BetaDistribution
from hypothesis_tests import ttest_1samp, confidence_interval, anova_oneway

s = DescriptiveStats([2, 4, 4, 4, 5, 5, 7, 9])
s.mean(), s.median(), s.mode()          # (5.0, 4.5, [4])
print(s.summary())

b = BinomialDistribution(n=10, p=0.3)
round(b.pmf(3), 4)                       # 0.2668
round(b.cdf(3), 4)                       # 0.6496

n = NormalDistribution(mu=0, sigma=1)
round(n.cdf(1.96), 4)                    # 0.975

beta = BetaDistribution(2, 5)
beta.mean(), beta.sample(3, seed=1)

r = ttest_1samp([5.1, 4.9, 5.3, 5.0, 4.8], pop_mean=5.0)
r["conclusion"]                          # 'Fail to reject H0'

ci = confidence_interval([5.1, 4.9, 5.3, 5.0, 4.8, 5.2], confidence=0.95)
ci["lower"], ci["upper"]

anova_oneway([1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8])["p_value"]

# Covariance/correlation matrices (Matrix-valued)
dataset = [[1, 2, 3, 4, 5], [2, 4, 5, 4, 5], [5, 3, 2, 1, 1]]
cov = DescriptiveStats.covariance_matrix(dataset)
corr = DescriptiveStats.correlation_matrix(dataset)
```

---

## Design Notes

- **Exact CDFs, not numerical integration.** `Chi2Distribution`,
  `BetaDistribution`, and `GammaDistribution` compute their CDFs via the
  regularized incomplete gamma/beta functions from `special_functions.py`
  rather than trapezoidal quadrature. This was a deliberate rewrite —
  dramatically faster (O(iterations of a continued fraction) instead of
  O(n_steps) quadrature calls) and more accurate — and is guarded by
  regression tests (`test_cdf_exact_matches_scipy`,
  `test_cdf_speed_is_fast`) so it can't silently regress.
- **Log-space arithmetic wherever products of many/large terms would
  overflow.** `PoissonDistribution.pmf`, `BetaDistribution.pdf`, and
  `GammaDistribution.pdf` all compute in log-space
  (`math.lgamma`/`math.log`/`math.exp`) rather than evaluating
  `lam**k / k!`-style products directly, which would overflow for even
  moderately large `k`, `alpha`, or `beta`.
- **Poisson sampling underflow workaround.** Knuth's direct-simulation
  algorithm needs `exp(-lam)`, which underflows to exactly `0.0` in IEEE
  double precision above `lam ≈ 700` — this would make the algorithm loop
  effectively forever rather than raise. `PoissonDistribution` exploits
  the infinite divisibility of the Poisson distribution: it decomposes
  `Poisson(lam)` into a sum of independent `Poisson(lam/m)` draws with
  `m` chosen to keep each chunk's rate at or below a safe threshold
  (`_KNUTH_SAFE_LAMBDA = 20.0`). Covered by a regression test
  (`test_large_lambda_sampling_does_not_hang`).
- **`bool` is rejected everywhere a numeric parameter is expected** —
  distribution parameters, dataset elements, discrete `k`/`n` — for the
  same reason as `Vector`/`Matrix`: `bool` is an `int` subclass in
  Python, and silently accepting `True`/`False` almost always indicates
  an upstream bug.
- **NaN is rejected, not propagated, in `DescriptiveStats`.** Unlike
  `Vector`, which lets NaN propagate through arithmetic, sorting-based
  statistics (median, percentile, mode) are undefined in the presence of
  NaN, since any comparison against NaN is `False` in IEEE-754. Rather
  than silently produce a wrong answer, the constructor validates
  up front and raises `ValueError` with an explanation.
- **Skewness/kurtosis of constant data return `0.0`, not an error.** A
  zero-variance dataset makes the standard formulas divide by zero;
  by convention, skewness and (excess) kurtosis of a degenerate
  distribution are taken to be `0.0` rather than raising, since "no
  asymmetry" and "no excess peakedness" are the mathematically sensible
  defaults for a point mass.
- **Undefined moments raise rather than return `NaN`.** `TDistribution`
  (`df <= 1`) and `FDistribution` (`mean` for `df2 <= 2`, `variance` for
  `df2 <= 4`) raise `ValueError` for parameter ranges where the
  moment is mathematically undefined or infinite-in-a-different-sense,
  rather than silently returning `NaN` or `inf` without explanation —
  except `TDistribution.variance()`, which does correctly return
  `math.inf` for `1 < df <= 2`, since that case has a well-defined
  (infinite) answer, unlike `df <= 1`.
- **Every hypothesis test returns a plain `dict`.** This keeps results
  trivially inspectable, loggable, and serializable without a custom
  result-object hierarchy — a deliberate simplicity trade-off consistent
  with the rest of this "from scratch" curriculum.
- **ANOVA and Welch's t-test handle degenerate variance explicitly**
  rather than letting a division by zero raise unhelpfully: identical
  groups yield `F=NaN, p=NaN`; zero within-group variance with nonzero
  between-group variance yields `F=inf, p=0.0` (perfect separation is
  still a meaningful, decisive result).
- **`covariance_matrix`/`correlation_matrix` exploit symmetry**, computing
  each off-diagonal pair once instead of twice, and set the correlation
  diagonal to exactly `1.0` by definition rather than computing it (which
  would otherwise require a zero-variance special case).

---

## Test Coverage

Each library file has a matching `tests/test_*.py` that cross-checks
numeric correctness against SciPy/NumPy (`scipy.stats`, `scipy.special`,
`numpy.mean`/`var`/`percentile`/`cov`/`corrcoef`) in addition to covering
construction/validation, edge cases (zero variance, degenerate
parameters, boundary values), and reproducibility (seeded sampling).
Notable regression tests: exact vs. trapezoidal CDF equivalence and speed
for Beta/Gamma, and the large-`lambda` Poisson sampling performance fix.

Run the full suite with verbose output:

```bash
pytest tests/ -v
```

---

## Roadmap Context

`Probability/` is the third module in **Phase 1: Math Foundations**,
building on both `Vectors/vector.py` (indirectly, via `Matrix`) and
`Matrix/matrix.py` (directly, for covariance/correlation matrices). It
carries forward the same conventions established there — explicit
exception hierarchy, boolean/NaN rejection where relevant, SciPy/NumPy
confined to tests, full docstring + type-hint coverage — while
introducing the statistical primitives (distributions, hypothesis
testing) that later phases (e.g. Bayesian methods, statistical ML
algorithms) will depend on.
