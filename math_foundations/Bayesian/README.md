# Bayesian — Pure Python Bayesian Inference Library

## Overview

The `Bayesian/` package is a pure Python implementation of Bayesian inference primitives — conjugate models, sampling algorithms, and posterior analysis utilities — built from scratch as part of the *Engineering Redemption Arc*, a structured 60-project ML engineering curriculum in the [`ml-implementation-from-scratch`](https://github.com/) repository.

The package covers the full Bayesian workflow: specifying priors, updating beliefs with data, computing posteriors analytically (conjugate models) or numerically (grid approximation, MCMC), and summarizing results with credible intervals, HDI, Bayes factors, and convergence diagnostics. All computations use Python's standard library only — no NumPy, SciPy, or PyMC.

---

## Project Structure

```
ml-implementation-from-scratch/
├── Vectors/
│   └── vector.py                  # Vector primitive (transitive dependency)
├── Matrix/
│   └── matrix.py                  # Matrix primitive (transitive dependency)
├── Probability/
│   ├── continuous.py              # NormalDistribution, BetaDistribution, GammaDistribution
│   ├── descriptive_stats.py       # DescriptiveStats (used in inference.py)
│   └── ...
└── Bayesian/
    ├── conjugate.py               # Analytic conjugate update models
    ├── inference.py               # Posterior summaries, HDI, Bayes factors, diagnostics
    └── samplers.py                # Grid approximation, rejection sampling, MCMC
```

Each file resolves its parent directory at runtime. The folder must be named `Bayesian/` and sit alongside `Probability/` and `Matrix/` for imports to succeed.

---

## Dependencies

| Requirement | Detail |
|---|---|
| Python | 3.10+ |
| `math`, `random`, `os`, `sys` | Standard library only |
| `Probability/continuous.py` | `NormalDistribution`, `BetaDistribution`, `GammaDistribution` |
| `Probability/descriptive_stats.py` | `DescriptiveStats` — used in `inference.py` |

No `pip install` required.

---

## Installation

Ensure the folder layout above is intact. Run scripts from the repository root, or insert the root into `sys.path`:

```python
import sys
sys.path.insert(0, "/path/to/ml-implementation-from-scratch")

from Bayesian.conjugate import BetaBinomial, GammaPoisson, GaussianGaussian
from Bayesian.inference import credible_interval, hdi, posterior_summary, bayes_factor
from Bayesian.samplers import grid_approximation, rejection_sampling, metropolis_hastings
```

---

## Module Reference

---

### `conjugate.py` — Analytic Conjugate Models

All three classes implement the same interface: `update()`, `posterior_mean()`, `posterior_std()`, `map_estimate()`, and `credible_interval()`. The posterior is stored as a distribution object in `.posterior` and updates in place on each call to `update()`.

---

#### `BetaBinomial(alpha_prior, beta_prior)`

Models a latent success probability `p` with a Beta prior, updated by Binomial observations. Posterior is Beta(α + k, β + n − k).

```python
from Bayesian.conjugate import BetaBinomial

model = BetaBinomial(alpha_prior=1.0, beta_prior=1.0)   # uniform prior

model.update(n_trials=20, k_success=14)
model.update(n_trials=10, k_success=7)    # sequential updates accumulate

model.posterior_mean()      # float
model.posterior_std()       # float
model.map_estimate()        # (alpha - 1) / (alpha + beta - 2); handles boundary cases
model.credible_interval(prob=0.95)
# {prob, lower, upper, mean, std}
```

`update()` validates that `k_success <= n_trials` and that both are non-negative integers. The `.posterior` attribute is always a live `BetaDistribution` instance.

---

#### `GammaPoisson(alpha_prior, beta_prior)`

Models a latent event rate `λ` with a Gamma prior (rate parameterization), updated by Poisson observations. Posterior is Gamma(α + Σk, β + Σt).

```python
from Bayesian.conjugate import GammaPoisson

model = GammaPoisson(alpha_prior=2.0, beta_prior=1.0)

model.update(k_events=5, t_exposure=1.0)   # 5 events over 1 unit of time
model.update(k_events=8, t_exposure=2.0)   # 8 events over 2 units

model.posterior_mean()
model.posterior_std()
model.map_estimate()        # (alpha - 1) / beta if alpha >= 1, else 0.0
model.credible_interval(prob=0.95)
# {prob, lower, upper, mean, std}
```

`t_exposure` defaults to `1.0`. `k_events` can be a float (for fractional counts or aggregated data).

---

#### `GaussianGaussian(prior_mu, prior_sigma, obs_sigma)`

Models an unknown mean `μ` with a Gaussian prior, given known observation noise `obs_sigma`. Posterior is computed via precision-weighted update.

```python
from Bayesian.conjugate import GaussianGaussian

model = GaussianGaussian(prior_mu=0.0, prior_sigma=1.0, obs_sigma=0.5)

model.update([2.1, 1.9, 2.3, 2.0])   # update with a batch of observations
model.update([2.2, 2.4])              # sequential batches accumulate

model.posterior_mean()     # = map_estimate() for Gaussian
model.posterior_std()
model.map_estimate()
model.credible_interval(prob=0.95)
# {prob, lower, upper, mean, std}
# Interval computed via Normal PPF (exact, not sampled)
```

`obs_sigma` is fixed at construction and does not update. All three sigmas must be positive.

---

### `inference.py` — Posterior Analysis Utilities

Standalone functions for analyzing posterior samples or grid-based posteriors. Imports `DescriptiveStats` for sample statistics.

---

#### `map_estimate(grid, posterior_values)`

Returns the grid point with the highest posterior value.

```python
from Bayesian.inference import map_estimate

theta_map = map_estimate(grid, posterior_values)   # float
```

---

#### `credible_interval(samples, prob=0.95)`

Equal-tailed credible interval computed from sample percentiles.

```python
from Bayesian.inference import credible_interval

ci = credible_interval(samples, prob=0.95)
# {prob, lower, upper}
```

---

#### `hdi(samples, prob=0.95)`

Highest Density Interval — the shortest interval containing `prob` of the posterior mass. Preferable to equal-tailed CI for skewed or multimodal posteriors.

```python
from Bayesian.inference import hdi

result = hdi(samples, prob=0.95)
# {prob, lower, upper, width}
```

Implemented via a linear scan over the sorted sample array. Requires at least 2 samples.

---

#### `posterior_summary(samples, ci_prob=0.95)`

Computes a full summary of a posterior sample in one call.

```python
from Bayesian.inference import posterior_summary

summary = posterior_summary(samples, ci_prob=0.95)
# {
#     "mean": float,
#     "std": float,
#     "median": float,
#     "mode_approx": float,     # KDE-based mode estimate
#     "ci_95": (lower, upper),  # equal-tailed
#     "hdi_95": (lower, upper), # highest density interval
#     "n_samples": int
# }
```

The mode is estimated via a Gaussian KDE with Silverman's bandwidth rule (`h = σ × (4/3n)^0.2`). Requires at least 2 samples.

---

#### `log_marginal_likelihood(grid, log_prior_values, log_likelihood_values)`

Estimates the log marginal likelihood by trapezoidal integration over a uniform grid, using the log-sum-exp trick for numerical stability.

```python
from Bayesian.inference import log_marginal_likelihood

log_ml = log_marginal_likelihood(grid, log_prior_values, log_likelihood_values)
# float
```

Grid must be uniformly spaced and have at least 2 points.

---

#### `bayes_factor(log_ml1, log_ml2)`

Computes the Bayes factor K = exp(log_ml1 − log_ml2) and interprets it on the Jeffreys scale.

```python
from Bayesian.inference import bayes_factor

result = bayes_factor(log_ml1, log_ml2)
# {
#     "K": float,
#     "log_K": float,
#     "evidence": str,    # e.g. "Strong for M1", "Decisive for M2"
#     "log_ml1": float,
#     "log_ml2": float
# }
```

Jeffreys scale thresholds: K > 100 → Decisive, K > 10 → Strong, K > 3.2 → Substantial, K > 1 → Weak (and symmetric for M2).

---

#### `sequential_update(model, data_stream)`

Applies a stream of batches to any conjugate model, recording a snapshot of the posterior after each step.

```python
from Bayesian.inference import sequential_update
from Bayesian.conjugate import BetaBinomial

model = BetaBinomial(1.0, 1.0)
stream = [(10, 6), (15, 9), (20, 13)]   # (n_trials, k_success) tuples

snapshots = sequential_update(model, stream)
# [
#     {"step": 1, "posterior_mean": float, "posterior_std": float},
#     {"step": 2, ...},
#     ...
# ]
```

Tuple batches call `model.update(*batch)` — compatible with `BetaBinomial` and `GammaPoisson`. List batches call `model.update(batch)` — compatible with `GaussianGaussian`.

---

#### `gelman_rubin(chains)`

Computes the Gelman-Rubin R̂ convergence diagnostic for multiple MCMC chains. Values close to 1.0 indicate convergence; values above 1.1 suggest the chains have not mixed.

```python
from Bayesian.inference import gelman_rubin

R_hat = gelman_rubin([chain1, chain2, chain3])   # float
```

Requires at least 2 chains, each with at least 2 samples. Chains are trimmed to the length of the shortest chain before computation.

---

### `samplers.py` — Sampling Algorithms

---

#### `grid_approximation(prior_fn, likelihood_fn, grid)`

Computes a normalized discrete posterior over a uniform parameter grid.

```python
from Bayesian.samplers import grid_approximation
import math

grid = [i / 100 for i in range(1, 100)]   # theta in (0.01, 0.99)

def prior_fn(theta):
    return 1.0   # uniform

def likelihood_fn(theta):
    k, n = 14, 20
    return (theta ** k) * ((1 - theta) ** (n - k))

result = grid_approximation(prior_fn, likelihood_fn, grid)
# {
#     "grid": list,
#     "posterior": list,             # normalized probabilities summing to 1
#     "log_unnorm": list,
#     "normalisation_constant": float
# }
```

Computation is done in log-space with a max-subtraction trick to avoid underflow. Grid must be uniformly spaced. Both `prior_fn` and `likelihood_fn` must return positive values; zero or negative values are treated as log(−∞).

---

#### `rejection_sampling(target_fn, proposal_sampler, proposal_fn, M, n_samples, max_iter=1_000_000)`

Draws exact samples from a target distribution using rejection sampling.

```python
from Bayesian.samplers import rejection_sampling
import random, math

def target_fn(x):
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)   # standard normal

def proposal_sampler():
    return random.uniform(-4, 4)

def proposal_fn(x):
    return 1 / 8   # uniform on [-4, 4]

M = 4.0   # upper bound: target_fn(x) <= M * proposal_fn(x) for all x

result = rejection_sampling(target_fn, proposal_sampler, proposal_fn, M=M, n_samples=500)
# {
#     "samples": list,
#     "n_accepted": int,
#     "n_rejected": int,
#     "acceptance_rate": float
# }
```

Raises `RuntimeError` if `max_iter` is reached before collecting `n_samples`. The acceptance rate equals `1/M` for an optimal envelope — low rates indicate `M` is too loose.

---

#### `metropolis_hastings(log_posterior_fn, init, n_samples, step_size=0.1, burn_in=500)`

Draws samples from an unnormalized posterior using the Metropolis-Hastings algorithm with a Gaussian random walk proposal.

```python
from Bayesian.samplers import metropolis_hastings
import math

def log_posterior_fn(theta):
    if theta <= 0 or theta >= 1:
        return float("-inf")
    return 14 * math.log(theta) + 6 * math.log(1 - theta)   # Beta(15, 7) unnormalized

result = metropolis_hastings(
    log_posterior_fn=log_posterior_fn,
    init=0.5,
    n_samples=2000,
    step_size=0.1,
    burn_in=500
)
# {
#     "samples": list,          # length = n_samples (burn-in excluded)
#     "acceptance_rate": float,
#     "n_samples": int,
#     "burn_in": int,
#     "step_size": float,
#     "init": float
# }
```

The acceptance step is computed entirely in log-space to avoid overflow. Burn-in samples are discarded automatically. A target acceptance rate of 20–50% is typical for a 1D random walk; tune `step_size` accordingly.

---

#### `effective_sample_size(samples)`

Estimates the effective sample size (ESS) of an MCMC chain by summing the autocorrelation function until it drops below 0.05.

```python
from Bayesian.samplers import effective_sample_size

ess = effective_sample_size(samples)   # float
```

Returns `float(n)` for chains shorter than 4 samples. ESS is clamped to `[1, n]`. Anti-correlation is handled correctly by truncating on `|ρ|` rather than `ρ`.

---

## Example Session

```python
import math
import random
from Bayesian.conjugate import BetaBinomial, GammaPoisson, GaussianGaussian
from Bayesian.inference import (
    posterior_summary, hdi, bayes_factor,
    log_marginal_likelihood, sequential_update, gelman_rubin
)
from Bayesian.samplers import (
    grid_approximation, metropolis_hastings, effective_sample_size
)

# --- Conjugate: Beta-Binomial ---
model = BetaBinomial(alpha_prior=1.0, beta_prior=1.0)
model.update(n_trials=30, k_success=21)
print(model.credible_interval(0.95))

# --- Conjugate: Gamma-Poisson ---
rate_model = GammaPoisson(alpha_prior=2.0, beta_prior=1.0)
rate_model.update(k_events=15, t_exposure=3.0)
print(rate_model.posterior_mean())

# --- Conjugate: Gaussian-Gaussian ---
mu_model = GaussianGaussian(prior_mu=0.0, prior_sigma=1.0, obs_sigma=0.5)
mu_model.update([1.8, 2.1, 2.0, 1.95, 2.05])
print(mu_model.credible_interval(0.90))

# --- Grid approximation ---
grid = [i / 200 for i in range(1, 200)]
result = grid_approximation(
    prior_fn=lambda theta: 1.0,
    likelihood_fn=lambda theta: (theta ** 21) * ((1 - theta) ** 9),
    grid=grid
)

# --- MCMC ---
def log_post(theta):
    if not (0 < theta < 1):
        return float("-inf")
    return 21 * math.log(theta) + 9 * math.log(1 - theta)

mcmc = metropolis_hastings(log_post, init=0.5, n_samples=5000, step_size=0.08)
samples = mcmc["samples"]
print(f"Acceptance rate: {mcmc['acceptance_rate']:.2f}")
print(f"ESS: {effective_sample_size(samples):.1f}")
print(posterior_summary(samples))

# --- Convergence diagnostic ---
chains = [
    metropolis_hastings(log_post, init=random.random(), n_samples=1000)["samples"]
    for _ in range(3)
]
print(f"R-hat: {gelman_rubin(chains):.4f}")

# --- Bayes factor ---
grid_bf = [i / 100 for i in range(1, 100)]
log_ml1 = log_marginal_likelihood(
    grid_bf,
    [math.log(1.0)] * len(grid_bf),                              # uniform prior M1
    [21 * math.log(t) + 9 * math.log(1 - t) for t in grid_bf]
)
log_ml2 = log_marginal_likelihood(
    grid_bf,
    [math.log(6 * t * (1 - t)) for t in grid_bf],               # Beta(2,2) prior M2
    [21 * math.log(t) + 9 * math.log(1 - t) for t in grid_bf]
)
print(bayes_factor(log_ml1, log_ml2))
```

---

## Error Reference

| Situation | Exception |
|---|---|
| Non-positive `alpha_prior` or `beta_prior` | `ValueError` |
| Non-positive `prior_sigma` or `obs_sigma` in GaussianGaussian | `ValueError` |
| `k_success > n_trials` in BetaBinomial | `ValueError` |
| Non-integer or negative `n_trials`/`k_success` | `ValueError` |
| Negative `k_events` or non-positive `t_exposure` | `ValueError` |
| Empty or non-list `data` in GaussianGaussian | `ValueError` / `TypeError` |
| `prob` outside `(0, 1)` in `credible_interval`/`hdi` | `ValueError` |
| Fewer than 2 samples for `hdi`/`posterior_summary` | `ValueError` |
| Empty `grid` or `samples` | `ValueError` |
| Non-uniform grid in `grid_approximation` or `log_marginal_likelihood` | `ValueError` |
| Posterior zero everywhere on grid | `ValueError` |
| `M <= 0` in rejection sampling | `ValueError` |
| `max_iter` exceeded in rejection sampling | `RuntimeError` |
| `log_posterior_fn` raises at `init` in MH | `ValueError` |
| Non-positive `step_size` or `n_samples` in MH | `ValueError` |
| Fewer than 2 chains or samples in `gelman_rubin` | `ValueError` |
| Mismatched lengths in `log_marginal_likelihood` | `ValueError` |
| Wrong batch type in `sequential_update` | `TypeError` |
| Tuple batch not length 2 in `sequential_update` | `ValueError` |

---

## Design Notes

- **Conjugate models update in place:** Each call to `.update()` mutates `self.alpha`/`self.beta` (or `self.prior_mu`/`self.prior_var`) and replaces `.posterior` with a fresh distribution instance. This means `.posterior` always reflects the cumulative evidence, and sequential updates require no external state management.
- **Log-space throughout:** Grid approximation, MH acceptance, and `log_marginal_likelihood` all operate in log-space. The log-sum-exp trick (`max_log` subtraction before `sum(exp(...))`) prevents underflow on narrow or peaked posteriors.
- **KDE mode in `posterior_summary`:** The mode is approximated using a full O(n²) Gaussian KDE pass over sorted samples — accurate for smooth unimodal posteriors but expensive for very large chains. Silverman's rule is used for bandwidth.
- **ESS truncation heuristic:** `effective_sample_size` stops accumulating autocorrelations when `|ρ| < 0.05`. This truncation on absolute value (not signed value) correctly handles anti-correlated chains, where signed truncation would over-estimate ESS.
- **MH is univariate:** `metropolis_hastings` operates on a single scalar parameter. Extending to multivariate posteriors requires replacing the Gaussian random walk with a multivariate proposal — not implemented here.
- **Grid approximation is exact up to grid resolution:** For conjugate models, the analytic posterior in `conjugate.py` is preferable. Grid approximation is most useful for non-conjugate likelihoods or custom priors.

---

## Roadmap Context

This package is a foundational project in the Engineering Redemption Arc curriculum. It depends on:

- **`Vectors/vector.py`** — transitive dependency via `Matrix/`
- **`Matrix/matrix.py`** — transitive dependency via `Probability/descriptive_stats.py`
- **`Probability/`** — direct dependency for `BetaDistribution`, `GammaDistribution`, `NormalDistribution`, and `DescriptiveStats`

It underpins:

- **Phase 2 ML models** (Projects 11–25) — Bayesian linear regression, Naive Bayes classifiers, and probabilistic model evaluation all build directly on conjugate updates and MCMC sampling implemented here.
