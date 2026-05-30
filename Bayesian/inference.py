import os
import sys
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from Probability.descriptive_stats import DescriptiveStats


def map_estimate(grid: list, posterior_values: list) -> float:
    """
    Return the grid point with the highest posterior density.
    """
    if len(grid) == 0:
        raise ValueError("grid must be non-empty")
    if len(grid) != len(posterior_values):
        raise ValueError("grid and posterior_values must have the same length")
    max_idx = posterior_values.index(max(posterior_values))
    return grid[max_idx]


def credible_interval(samples: list, prob: float = 0.95) -> dict:
    """
    Equal-tailed credible interval (percentile-based) using linear interpolation.
    """
    if not samples:
        raise ValueError("samples list is empty")
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must be strictly between 0 and 1")

    stats = DescriptiveStats(samples)
    alpha = (1.0 - prob) / 2.0
    lower = stats.percentile(100.0 * alpha)
    upper = stats.percentile(100.0 * (1.0 - alpha))

    return {
        "prob": prob,
        "lower": lower,
        "upper": upper,
    }


def hdi(samples: list, prob: float = 0.95) -> dict:
    """
    Highest Posterior Density interval approximated from samples.
    """
    if not samples:
        raise ValueError("samples list is empty")
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to estimate HDI")
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must be strictly between 0 and 1")

    sorted_s = sorted(samples)
    n = len(sorted_s)
    interval_size = max(1, int(math.ceil(prob * n)))  # number of points in the interval
    if interval_size >= n:
        return {"prob": prob, "lower": sorted_s[0], "upper": sorted_s[-1], "width": sorted_s[-1] - sorted_s[0]}

    best_width = float("inf")
    best_lower = sorted_s[0]
    best_upper = sorted_s[-1]

    for i in range(n - interval_size + 1):
        low = sorted_s[i]
        high = sorted_s[i + interval_size - 1]
        width = high - low
        if width < best_width:
            best_width = width
            best_lower = low
            best_upper = high

    return {
        "prob": prob,
        "lower": best_lower,
        "upper": best_upper,
        "width": best_width,
    }


def posterior_summary(samples: list, ci_prob: float = 0.95) -> dict:
    """
    Compute mean, std, median, approximate mode (via KDE), credible and HDI intervals.
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples for summary statistics")

    n = len(samples)
    stats = DescriptiveStats(samples)

    mean = stats.mean()
    std = stats.std(ddof=0)        # population std (we treat samples as full posterior)
    median = stats.median()

    # Approximate mode using KDE (same bandwidth as before)
    bandwidth = std * (4.0 / (3.0 * n)) ** 0.2 if std > 0 else 1e-6
    kde_vals = _kde_values(sorted(samples), bandwidth)
    mode_idx = max(range(n), key=lambda i: kde_vals[i])
    mode_approx = samples[mode_idx]

    ci = credible_interval(samples, ci_prob)
    hdi_res = hdi(samples, ci_prob)

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "mode_approx": mode_approx,
        f"ci_{int(ci_prob * 100)}": (ci["lower"], ci["upper"]),
        f"hdi_{int(ci_prob * 100)}": (hdi_res["lower"], hdi_res["upper"]),
        "n_samples": n,
    }


def _kde_values(sorted_samples: list, bandwidth: float) -> list:
    """Gaussian KDE evaluated at each sample point (for mode finding)."""
    n = len(sorted_samples)
    if bandwidth <= 0.0:
        bandwidth = 1e-6
    densities = []
    inv_denom = 1.0 / (n * bandwidth * math.sqrt(2.0 * math.pi))
    for x in sorted_samples:
        # sum of gaussian kernels: exp(-0.5 * ((x-xi)/bw)^2)
        kernel_sum = sum(
            math.exp(-0.5 * ((x - xi) / bandwidth) ** 2)
            for xi in sorted_samples
        )
        densities.append(kernel_sum * inv_denom)
    return densities


def log_marginal_likelihood(
    grid: list,
    log_prior_values: list,
    log_likelihood_values: list,
) -> float:
    """
    Compute log marginal likelihood using a rectangular integration rule.
    Assumes uniformly spaced grid.
    """
    if len(grid) < 2:
        raise ValueError("grid must have at least 2 points")
    if len(grid) != len(log_prior_values) or len(grid) != len(log_likelihood_values):
        raise ValueError("grid, log_prior_values, log_likelihood_values must have same length")

    delta = (grid[-1] - grid[0]) / (len(grid) - 1)
    # numerically stable log-sum-exp
    log_vals = [lp + ll for lp, ll in zip(log_prior_values, log_likelihood_values)]
    max_lv = max(log_vals)
    log_integral = max_lv + math.log(
        sum(math.exp(lv - max_lv) for lv in log_vals) * delta
    )
    return log_integral


def bayes_factor(log_ml1: float, log_ml2: float) -> dict:
    """
    Bayes factor from two log marginal likelihoods.
    Interpretation follows Jeffreys (1961).
    """
    log_K = log_ml1 - log_ml2
    try:
        K = math.exp(log_K)
    except OverflowError:
        K = float("inf")

    # Jeffreys' scale
    if K == float("inf") or K > 100:
        evidence = "Decisive for M1"
    elif K > 10:
        evidence = "Strong for M1"
    elif K > 3.2:
        evidence = "Substantial for M1"
    elif K > 1:
        evidence = "Weak for M1"
    elif K == 1.0:
        evidence = "No preference"
    elif K > (1.0 / 3.2):
        evidence = "Weak for M2"
    elif K > 0.1:
        evidence = "Substantial for M2"
    elif K > 0.01:
        evidence = "Strong for M2"
    else:
        evidence = "Decisive for M2"

    return {
        "K": K,
        "log_K": log_K,
        "evidence": evidence,
        "log_ml1": log_ml1,
        "log_ml2": log_ml2,
    }


def sequential_update(model, data_stream: list) -> list:
    """
    Update a conjugate model sequentially and return snapshots after each batch.
    model must have an update() method and posterior_mean/std.
    data_stream: list of batches. Each batch can be:
    - a tuple of two numbers (e.g. (n_trials, k_successes)) for BetaBinomial/GammaPoisson
    - a list of observations for GaussianGaussian
    """
    snapshots = []
    for i, batch in enumerate(data_stream):
        if isinstance(batch, tuple) and len(batch) == 2:
            model.update(*batch)
        else:
            model.update(batch)
        snapshots.append({
            "step": i + 1,
            "posterior_mean": model.posterior_mean(),
            "posterior_std": model.posterior_std(),
        })
    return snapshots


def gelman_rubin(chains: list) -> float:
    """
    Gelman‑Rubin R‑hat convergence diagnostic.
    chains: list of chains (each chain is a list of samples).
    """
    if len(chains) < 2:
        raise ValueError("Need at least 2 chains for Gelman‑Rubin diagnostic")

    # Use the shortest chain length (common practice)
    n = min(len(c) for c in chains)
    if n < 2:
        raise ValueError("Each chain must have at least 2 samples")

    m = len(chains)
    # discard burn‑in? Not here; we assume all samples are kept.
    trimmed = [c[:n] for c in chains]
    chain_means = [sum(c) / n for c in trimmed]
    grand_mean = sum(chain_means) / m

    # between‑chain variance
    B = n * sum((cm - grand_mean) ** 2 for cm in chain_means) / (m - 1)

    # within‑chain variances
    W = sum(
        sum((x - chain_means[j]) ** 2 for x in trimmed[j]) / (n - 1)
        for j in range(m)
    ) / m

    if W == 0.0:
        return float("inf")

    var_plus = ((n - 1) * W + B) / n
    R_hat = math.sqrt(var_plus / W)
    return R_hat
