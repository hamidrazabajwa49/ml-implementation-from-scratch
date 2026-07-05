# Bayesian — Conjugate Models, Monte Carlo Samplers & Posterior Inference (From Scratch)

**Phase 1: Math Foundations**.

A dependency-free, pure-Python Bayesian inference toolkit: closed-form
conjugate-prior update models, Monte Carlo approximation tools (grid
approximation, rejection sampling, Metropolis-Hastings), and sample-/
grid-based posterior summary utilities (credible intervals, HDI, log
marginal likelihood, Bayes factors, MCMC convergence diagnostics).
NumPy/SciPy are used only in the test suite, as correctness oracles and
RNGs.

---

## Overview

| File | Provides |
|---|---|
| `conjugate.py` | `ConjugateModel` base class + `BetaBinomial`, `GammaPoisson`, `GaussianGaussian` closed-form conjugate update models |
| `samplers.py` | `grid_approximation`, `rejection_sampling`, `metropolis_hastings`, `effective_sample_size` |
| `inference.py` | `map_estimate`, `credible_interval`, `hdi`, `posterior_summary`, `log_marginal_likelihood`, `bayes_factor`, `sequential_update`, `gelman_rubin` |

`conjugate.py` builds on `continuous.py`'s `NormalDistribution` /
`BetaDistribution` / `GammaDistribution` (each conjugate model's
posterior *is* one of those distribution objects). `inference.py` builds
on `descriptive_stats.py`'s `DescriptiveStats` for validated summary
statistics, and can drive any `ConjugateModel` via `sequential_update`.
`samplers.py` is self-contained (stdlib `random`/`math`/`logging` only).

---

## Project Structure

```
math_foundations/
└── Bayesian/
    ├── conjugate.py
    ├── samplers.py
    ├── inference.py
    ├── tests/
    │   ├── test_conjugate.py
    │   ├── test_samplers.py
    │   └── test_inference.py
    └── README.md
```

`conjugate.py` imports from `Probability.continuous`; `inference.py`
imports from `Probability.descriptive_stats`; both via relative
`sys.path` inserts. `Vectors`, `Matrix`, `Probability`, and `Bayesian`
must all live side-by-side under `math_foundations/`.

---

## Dependencies

| Component | Requires |
|---|---|
| Library files | Python 3.8+ standard library only (`math`, `random`, `logging`, `abc`, `typing`) + `Probability.continuous` / `Probability.descriptive_stats` (sibling module) |
| `tests/*.py` | `pytest`, `numpy`, `scipy` (test-only, for regression checks and RNG) |

Install test dependencies:

```bash
pip install pytest numpy scipy
```

Run the full suite from `math_foundations/Bayesian/`:

```bash
pytest tests/ -v
```

---

## Module Reference

### `conjugate.py`

```python
class ConjugateModel(ABC):
    posterior: BetaDistribution | GammaDistribution | NormalDistribution
    def update(self, *args, **kwargs) -> None       # abstract, model-specific
    def map_estimate(self) -> float                  # abstract, model-specific
    def posterior_mean(self) -> float                 # self.posterior.mean()
    def posterior_std(self) -> float                  # sqrt(self.posterior.variance())
    def credible_interval(self, prob: float = 0.95) -> Dict[str, float]
```
`credible_interval` is equal-tailed, computed via `self.posterior.ppf` at the two tail quantiles. Returns `{"prob", "lower", "upper", "mean", "std"}`. Raises `ValueError` if `prob` isn't strictly between 0 and 1. Implemented once in the base class in terms of any concrete model's `posterior` distribution object — subclasses only need to implement `update` and `map_estimate`.

```python
class BetaBinomial(ConjugateModel):
    def __init__(self, alpha_prior: float, beta_prior: float)
    def update(self, n_trials: int, k_success: int) -> None
    def map_estimate(self) -> float
```
Beta prior / Binomial likelihood, for an unknown success probability. `update` adds `k_success` to `alpha` and `n_trials - k_success` to `beta`; sequential updates are associative (equivalent to one combined update). `map_estimate` raises `ValueError` when `alpha <= 1` and `beta <= 1` (uniform or bimodal density — no unique finite mode exists).

```python
class GammaPoisson(ConjugateModel):
    def __init__(self, alpha_prior: float, beta_prior: float)
    def update(self, k_events: float, t_exposure: float = 1.0) -> None
    def map_estimate(self) -> float
```
Gamma prior / Poisson likelihood, for an unknown rate. `update` adds `k_events` to `alpha` and `t_exposure` to `beta`. `map_estimate` returns `(alpha-1)/beta` for `alpha >= 1`, else `0.0` (boundary mode).

```python
class GaussianGaussian(ConjugateModel):
    def __init__(self, prior_mu: float, prior_sigma: float, obs_sigma: float)
    def update(self, data: Sequence[Number]) -> None
    def map_estimate(self) -> float
```
Normal prior / Normal likelihood, for an unknown mean with known observation variance. `update` performs a precision-weighted batch update (`post_prec = prior_prec + n/obs_var`). `map_estimate` equals the posterior mean (mode = mean for a Normal).

All three constructors and `update` methods reject `bool` and non-finite/NaN inputs explicitly; `GaussianGaussian.update` additionally requires `data` to be a `list`/`tuple` (not a raw string) and rejects NaN/Inf elements.

### `samplers.py`

```python
grid_approximation(prior_fn, likelihood_fn, grid) -> dict
```
Discretized Bayesian posterior via a Riemann sum over a **strictly increasing, uniformly-spaced** grid, evaluated in log-space for numerical stability. Returns `{"grid", "posterior", "log_unnorm", "normalisation_constant"}`. Raises `ValueError` for fewer than 2 grid points, a non-increasing or non-uniform grid, or a posterior that's zero/underflows everywhere.

```python
rejection_sampling(target_fn, proposal_sampler, proposal_fn, M, n_samples, max_iter=1_000_000, bound_check_tol=1e-9) -> dict
```
Classic accept/reject sampling requiring `target_fn(x) <= M * proposal_fn(x)`. Bound violations are detected and logged once (not raised — a warning is enough to alert the caller without aborting a long-running job), since a violated bound means accepted samples no longer follow the target distribution. Returns `{"samples", "n_accepted", "n_rejected", "acceptance_rate"}`. Raises `ValueError` for non-positive `M`/`n_samples`/`max_iter`, `RuntimeError` if `max_iter` proposals are exhausted or `target_fn` itself raises.

```python
metropolis_hastings(log_posterior_fn, init, n_samples, step_size=0.1, burn_in=500) -> dict
```
Random-walk Metropolis-Hastings with a Gaussian proposal. Validates the initial log-posterior is neither NaN nor `-inf` before starting. Exceptions raised by `log_posterior_fn` on a *proposed* (non-initial) point are treated as `-inf` (auto-reject) rather than crashing the sampler. Returns `{"samples", "acceptance_rate", "n_samples", "burn_in", "step_size", "init"}` (acceptance rate is post-burn-in only).

```python
effective_sample_size(samples, max_lag=1000) -> float
```
Autocorrelation-based ESS: sums lag-k autocorrelations until `|rho_k| < 0.05` (truncating on absolute value handles anti-correlated chains), capped at `max_lag` lags to avoid O(n²) blow-up on long chains with slowly-decaying autocorrelation. Handles degenerate cases explicitly: constant sequences return `1.0`; a denominator pushed non-positive by extreme anti-correlation falls back to `1.0` rather than dividing by zero. Result is always clamped to `[1, len(samples)]`.

### `inference.py`

```python
map_estimate(grid, posterior_values) -> Number
```
Grid-search MAP: the grid point with the highest (possibly unnormalized) posterior value; ties return the first (lowest-index) occurrence. Raises `ValueError` for an empty grid, length mismatch, or negative/NaN posterior values.

```python
credible_interval(samples, prob=0.95) -> dict
```
Equal-tailed interval from empirical percentiles of `samples` (via `DescriptiveStats.percentile`). Returns `{"prob", "lower", "upper"}`.

```python
hdi(samples, prob=0.95) -> dict
```
Highest Density Interval — the shortest window containing `ceil(prob * n)` sorted sample points (Chen & Shao, 1999 linear-scan method). Returns `{"prob", "lower", "upper", "width"}`. Needs at least 2 samples.

```python
posterior_summary(samples, ci_prob=0.95, kde_grid_size=500) -> dict
```
One-stop summary: `{"mean", "std", "median", "mode_approx", "ci_<pct>", "hdi_<pct>", "n_samples"}`. The mode estimate uses a Gaussian KDE evaluated on a **fixed-size grid** (`_kde_mode`, Silverman's-rule-scaled bandwidth) rather than at every sample point — O(n · grid_size) instead of O(n²), which matters for MCMC chains in the tens of thousands.

```python
log_marginal_likelihood(grid, log_prior_values, log_likelihood_values) -> float
```
Model evidence via trapezoidal quadrature in log-space, using the log-sum-exp trick for numerical stability. Correctly supports **non-uniformly spaced** grids (interior points weighted by half the sum of their two neighboring intervals, endpoints by half their single adjacent interval) — a documented fix over a naive fixed-delta Riemann sum. Requires a strictly increasing grid of at least 2 points.

```python
bayes_factor(log_ml1, log_ml2) -> dict
```
`K = ML1/ML2` with a qualitative interpretation on Jeffreys' (1961) scale (`"Decisive"`/`"Strong"`/`"Substantial"`/`"Weak" for M1|M2`, or `"No preference"`). `OverflowError` from `exp(log_K)` is caught and reported as `K = inf`. Returns `{"K", "log_K", "evidence", "log_ml1", "log_ml2"}`.

```python
sequential_update(model, data_stream) -> List[dict]
```
Applies a stream of data batches to any object exposing `update`/`posterior_mean`/`posterior_std` (i.e. any `ConjugateModel`), snapshotting the posterior after each batch. A `tuple` batch is unpacked as positional args (`BetaBinomial`/`GammaPoisson`); a `list` batch is passed as a single argument (`GaussianGaussian`). Raises `TypeError` if `model` lacks the required methods or a batch is neither a tuple nor a list, `ValueError` if a tuple batch isn't length-2.

```python
gelman_rubin(chains) -> float
```
Multi-chain R-hat convergence diagnostic. Chains of unequal length are silently truncated to the shortest. Explicitly handles the zero-within-chain-variance edge case: `1.0` if chains also agree (`B == 0`, trivial perfect convergence), `inf` if they disagree (genuine non-convergence) — rather than always returning `inf` whenever `W == 0`. Raises `ValueError` for fewer than 2 chains, fewer than 2 samples per (truncated) chain, or NaN values.

---

## Example Session

```python
from conjugate import BetaBinomial, GammaPoisson, GaussianGaussian
from samplers import grid_approximation, metropolis_hastings, effective_sample_size
from inference import credible_interval, hdi, posterior_summary, gelman_rubin, sequential_update

# Conjugate update
model = BetaBinomial(alpha_prior=1.0, beta_prior=1.0)
model.update(n_trials=10, k_success=7)
round(model.posterior_mean(), 4)          # 0.6667
model.credible_interval(0.95)             # {'prob': 0.95, 'lower': ..., 'upper': ..., ...}

# Streaming updates
snapshots = sequential_update(BetaBinomial(1.0, 1.0), [(10, 7), (5, 2)])

# Grid approximation
grid = [i / 999 for i in range(1000)]
result = grid_approximation(lambda t: 1.0, lambda t: t**7 * (1 - t)**3, grid)

# MCMC
log_post = lambda theta: -0.5 * (theta - 2.0) ** 2
mcmc = metropolis_hastings(log_post, init=0.0, n_samples=2000, burn_in=200)
ess = effective_sample_size(mcmc["samples"])

# Posterior summaries
summary = posterior_summary(mcmc["samples"], ci_prob=0.95)
hdi_result = hdi(mcmc["samples"], prob=0.9)

# Convergence diagnostic across independent chains
chains = [metropolis_hastings(log_post, init=0.0, n_samples=1000)["samples"] for _ in range(4)]
gelman_rubin(chains)                      # ~1.0 if converged
```

---

## Design Notes

- **`ConjugateModel` centralizes the shared math, subclasses own the
  update rule.** `posterior_mean`, `posterior_std`, and
  `credible_interval` are implemented exactly once, against the
  `self.posterior` distribution object (a `continuous.py` distribution
  instance) — every conjugate model reuses the same, already-tested
  `ppf`/`mean`/`variance` machinery instead of re-deriving quantiles by
  hand. Only `update` (the conjugate arithmetic) and `map_estimate` (the
  mode, which differs per distribution family) are genuinely
  model-specific.
- **`BetaBinomial.map_estimate` raises rather than guessing.** For
  `alpha <= 1` and `beta <= 1` the Beta density is uniform or bimodal
  with divergent peaks at both 0 and 1 — there is no single finite mode.
  Returning `0.5` here would silently return the *antimode* (the point
  of lowest density) rather than a mode; raising `ValueError` with a
  clear explanation is the honest answer, and is guarded by a regression
  test.
- **Sequential (streaming) conjugate updates are associative.** Because
  the Beta-Binomial and Gamma-Poisson posteriors are simple parameter
  additions, updating in two batches gives exactly the same posterior as
  one combined update — an invariant directly tested in
  `test_sequential_updates_are_associative`.
- **`grid_approximation` validates its grid strictly.** A descending or
  non-monotonic grid would previously silently flip the sign of (or
  otherwise corrupt) the normalization constant; the function now
  explicitly requires strictly increasing, uniformly-spaced input and
  raises `ValueError` otherwise. Evaluation is done in log-space
  (`log(prior) + log(likelihood)`, shifted by the max before
  exponentiating) to avoid overflow/underflow when densities span a wide
  dynamic range.
- **`log_marginal_likelihood` supports non-uniform grids correctly.**
  An earlier version assumed uniform spacing (using a single fixed
  delta), which silently gave wrong answers on non-uniform grids.
  Trapezoidal weights are now computed per-point from actual neighbor
  spacing, and the log-sum-exp trick keeps the summation numerically
  stable even when log-values span a huge range. Both the uniform and
  non-uniform cases are covered by regression tests against a known
  analytic integral.
- **`rejection_sampling` detects (but doesn't abort on) an invalid
  envelope.** If `target_fn(x) > M * proposal_fn(x)` for any sampled
  `x`, the accepted samples no longer follow the target distribution —
  a serious correctness issue. Rather than raising mid-run (which would
  kill a long-running job over a single edge case) or silently producing
  a biased sample, this condition is logged once as a `WARNING`,
  sufficient to alert the caller.
- **Poisson underflow-style safety nets appear throughout.**
  `effective_sample_size`'s lag cap prevents O(n²) blow-up on long
  chains with slow-decaying autocorrelation; `_kde_mode`'s fixed grid
  size prevents the same for large posterior sample sets in
  `posterior_summary` — both are deliberate complexity bounds, not
  incidental optimizations, and both are covered by dedicated
  performance regression tests.
- **Degenerate MCMC diagnostics are handled explicitly, not left to
  divide-by-zero.** `effective_sample_size` falls back to `1.0` if
  extreme anti-correlation would otherwise push its denominator
  non-positive. `gelman_rubin` distinguishes "all chains are the same
  constant" (`R-hat = 1.0`, trivial convergence) from "all chains have
  zero internal variance but disagree with each other" (`R-hat = inf`,
  genuine non-convergence) — both zero-within-variance cases, but with
  opposite diagnostic meaning.
- **`metropolis_hastings` validates the *starting* point strictly but
  tolerates *proposal* failures.** A `NaN` or `-inf` initial
  log-posterior is a configuration error and raises immediately. A
  proposed point that causes `log_posterior_fn` to raise (e.g. out of a
  function's valid domain) is instead treated as `-inf` and rejected —
  a normal, expected part of exploring parameter space, not a fatal
  error.
- **`bool` and NaN are rejected wherever a numeric parameter is
  expected**, consistent with the rest of this curriculum (`Vector`,
  `Matrix`, `Probability`).

---

## Test Coverage

Each library file has a matching `tests/test_*.py`. `test_conjugate.py`
cross-checks posterior mean/std/credible intervals against
`scipy.stats.beta`/`gamma`/`norm` and verifies the conjugate-update
arithmetic and associativity directly. `test_samplers.py` and
`test_inference.py` validate against known analytic results (e.g. a
uniform-prior + Binomial likelihood recovering the Beta-Binomial
posterior mean via `grid_approximation`, or a flat log-likelihood
integrating to a standard normal's own normalization via
`log_marginal_likelihood`), plus edge cases (degenerate/constant
sequences, invalid parameters, exception handling) and performance
regressions (KDE mode estimation, ESS lag-capping).

Run the full suite with verbose output:

```bash
pytest tests/ -v
```

---

## Roadmap Context

`Bayesian/` is the fourth module in **Phase 1: Math Foundations**,
building on `Probability/continuous.py` (conjugate posteriors) and
`Probability/descriptive_stats.py` (sample-based summaries). It
completes the statistical foundation — distributions, hypothesis
testing, and now Bayesian inference — that later phases (Bayesian ML
models, probabilistic graphical models, MCMC-based algorithms) will
build directly on top of.
