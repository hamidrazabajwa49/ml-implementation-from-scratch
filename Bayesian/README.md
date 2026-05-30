# Bayesian Inference Module

Built from scratch in pure Python — no NumPy, no SciPy.
Part of a 60-project ML/AI foundations series.

---

## Overview

This module implements Bayesian inference from first principles: conjugate models with closed-form posteriors, numerical grid approximation, MCMC sampling, and model comparison utilities.

All distributions used here live in `../Probability/continuous.py` and follow a shared interface (`pdf`, `cdf`, `ppf`, `sample`, `mean`, `variance`).

---

## Project Structure

```
bayesian/
├── conjugate.py      # Closed-form conjugate pairs
├── samplers.py       # Grid approximation + MCMC
└── inference.py      # MAP, credible intervals, Bayes factor, sequential update

Probability/
├── continuous.py     # NormalDistribution, BetaDistribution, GammaDistribution
├── discrete.py       # BernoulliDistribution, BinomialDistribution, PoissonDistribution
├── distributions.py  # TDistribution, Chi2Distribution, FDistribution
├── hypothesis.py     # t-tests, ANOVA, chi-square tests
└── special_functions.py  # betainc, gammainc (Lentz continued fraction)
```

---

## Conjugate Models (`conjugate.py`)

Conjugate pairs give closed-form posteriors — no sampling required.
The posterior is the same family as the prior, updated analytically.

### BetaBinomial

**Use case:** estimating a probability `p` from binary outcomes (coin flips, click-through rates, conversion rates).

```
Prior:      θ ~ Beta(α, β)
Likelihood: k | θ ~ Binomial(n, θ)
Posterior:  θ | k ~ Beta(α + k, β + n - k)
```

```python
from bayesian.conjugate import BetaBinomial

# Uninformative prior: Beta(1, 1) = Uniform
model = BetaBinomial(alpha_prior=1, beta_prior=1)

# Observe 7 successes in 10 trials
model.update(n_trials=10, k_success=7)

print(model.posterior_mean())          # 0.727
print(model.credible_interval(0.95))   # {'lower': 0.43, 'upper': 0.93, ...}
```

**Sequential updating** — each posterior becomes the next prior:

```python
model.update(n_trials=20, k_success=14)   # add more data
model.update(n_trials=5,  k_success=4)    # and more
```

---

### GammaPoisson

**Use case:** estimating a rate `λ` from count data (events per unit time, requests per second, word frequencies).

```
Prior:      λ ~ Gamma(α, β)
Likelihood: k | λ ~ Poisson(λ · t)
Posterior:  λ | k ~ Gamma(α + k, β + t)
```

```python
from bayesian.conjugate import GammaPoisson

# Prior: expect ~2 events/hour (α=2, β=1)
model = GammaPoisson(alpha_prior=2, beta_prior=1)

# Observe 15 events over 5 hours
model.update(k_events=15, t_exposure=5.0)

print(model.posterior_mean())          # 2.33 events/hour
print(model.credible_interval(0.95))
```

---

### GaussianGaussian

**Use case:** estimating an unknown mean `μ` when the observation noise `σ` is known (signal processing, sensor calibration, A/B test means).

```
Prior:      μ ~ Normal(μ₀, σ₀)
Likelihood: x | μ ~ Normal(μ, σ_obs)
Posterior:  μ | x ~ Normal(μ_post, σ_post)
```

Posterior precision = prior precision + data precision:

```
1/σ_post² = 1/σ₀² + n/σ_obs²
μ_post    = (μ₀/σ₀² + n·x̄/σ_obs²) × σ_post²
```

```python
from bayesian.conjugate import GaussianGaussian

# Prior: mean is around 170cm ± 10cm
model = GaussianGaussian(prior_mu=170, prior_sigma=10, obs_sigma=5)

# Observe 8 measurements
data = [172, 168, 175, 171, 169, 174, 170, 173]
model.update(data)

print(model.posterior_mean())          # pulled toward data
print(model.credible_interval(0.95))
```

---

## Samplers (`samplers.py`)

For non-conjugate problems where closed-form posteriors don't exist.

### Grid Approximation

Discretizes the parameter space, evaluates `prior × likelihood` at each point, and normalizes.

```python
from bayesian.samplers import grid_approximation

posterior_grid = grid_approximation(
    prior_fn=lambda theta: beta_pdf(theta, 2, 2),
    likelihood_fn=lambda theta: binomial_likelihood(theta, n=10, k=7),
    grid=[ i/100 for i in range(1, 100) ]
)
# Returns list of (theta, probability) pairs
```

**Best for:** 1D or 2D parameter spaces. Exact (up to grid resolution). Exponentially expensive beyond 2D.

---

### Rejection Sampling

Samples from a target distribution using a proposal envelope.

```python
from bayesian.samplers import rejection_sampling

samples = rejection_sampling(
    target_fn=posterior_fn,
    proposal_fn=proposal_fn,
    proposal_sampler=sampler_fn,
    M=3.0,          # envelope constant: M × proposal ≥ target everywhere
    n_samples=1000
)
```

**Best for:** low-dimensional posteriors with known envelope. Acceptance rate drops sharply in high dimensions.

---

### Metropolis-Hastings

MCMC sampler — constructs a Markov chain whose stationary distribution is the target posterior.

```python
from bayesian.samplers import metropolis_hastings

samples = metropolis_hastings(
    log_posterior_fn=lambda theta: log_prior(theta) + log_likelihood(theta),
    initial=0.5,
    proposal_std=0.1,
    n_samples=5000,
    burn_in=500
)
```

**Key properties:**
- Only requires unnormalized log-posterior (avoids numerical underflow)
- `burn_in` samples discarded to reduce dependence on initialization
- `proposal_std` controls step size — tune so acceptance rate ≈ 20–50%

---

## Inference Utilities (`inference.py`)

### MAP Estimate

Maximum A Posteriori — the mode of the posterior. Point estimate that incorporates the prior.

```python
from bayesian.inference import map_estimate

theta_map = map_estimate(
    log_posterior_fn=log_posterior,
    grid=[ i/1000 for i in range(1, 1000) ]
)
```

### Credible Interval

The Bayesian analogue of a confidence interval. `P(lower < θ < upper | data) = prob`.

```python
from bayesian.inference import credible_interval

ci = credible_interval(samples, prob=0.95)
# {'lower': 0.51, 'upper': 0.83, 'mean': 0.67}
```

### Bayes Factor

Compares evidence for two models M1 vs M2.

```
BF = P(data | M1) / P(data | M2)
```

| BF value | Interpretation |
|---|---|
| BF > 100 | Decisive evidence for M1 |
| BF 30–100 | Very strong |
| BF 10–30 | Strong |
| BF 3–10 | Moderate |
| BF 1–3 | Anecdotal |

```python
from bayesian.inference import bayes_factor

bf = bayes_factor(log_marginal_M1, log_marginal_M2)
```

### Sequential Update

Prior → posterior → new prior as data arrives in batches.

```python
from bayesian.inference import sequential_update

model = BetaBinomial(1, 1)
for batch in data_stream:
    sequential_update(model, batch)
```

---

## Key Intuitions

**Why Bayesian?**
Frequentist: "Given H0, how surprising is this data?"
Bayesian: "Given this data, what should I believe about θ?"

**Prior choice matters most with small data.** As n → ∞, the likelihood dominates and prior washes out.

**Conjugate pairs are exact and fast.** Use them whenever the likelihood family matches. Fall back to MCMC only when needed.

**Credible intervals mean what people think confidence intervals mean.** `P(θ ∈ [a,b] | data) = 0.95` is a direct probability statement about θ.

---

## Dependencies

```
Python 3.10+   (standard library only)

Internal:
  Probability.continuous     → NormalDistribution, BetaDistribution, GammaDistribution
  Probability.special_functions → betainc, gammainc
```

No external packages required.

---

## Part of

**60-Day ML/AI Foundations Challenge**
Building every algorithm from scratch — linear algebra → statistics → ML models → deep learning → LLMs.

→ [View full roadmap](#) | [LinkedIn](#) | [Twitter/X](#)
