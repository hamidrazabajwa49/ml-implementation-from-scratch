import os
import sys
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Probability.descriptive_stats import DescriptiveStats


def map_estimate(grid: list, posterior_values: list) -> float:
    if len(grid) == 0:
        raise ValueError("grid must be non-empty")
    if len(grid) != len(posterior_values):
        raise ValueError("grid and posterior_values must have the same length")
    return grid[posterior_values.index(max(posterior_values))]


def credible_interval(samples: list, prob: float = 0.95) -> dict:
    if not samples:
        raise ValueError("samples list is empty")
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must be strictly between 0 and 1")
    stats = DescriptiveStats(samples)
    alpha = (1.0 - prob) / 2.0
    return {
        "prob": prob,
        "lower": stats.percentile(100.0 * alpha),
        "upper": stats.percentile(100.0 * (1.0 - alpha)),
    }


def hdi(samples: list, prob: float = 0.95) -> dict:
    if not samples:
        raise ValueError("samples list is empty")
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to estimate HDI")
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must be strictly between 0 and 1")

    sorted_s = sorted(samples)
    n = len(sorted_s)
    interval_size = max(1, int(math.ceil(prob * n)))

    if interval_size >= n:
        return {
            "prob": prob,
            "lower": sorted_s[0],
            "upper": sorted_s[-1],
            "width": sorted_s[-1] - sorted_s[0],
        }

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
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples for summary statistics")
    if not (0.0 < ci_prob < 1.0):
        raise ValueError("ci_prob must be strictly between 0 and 1")

    n = len(samples)
    stats = DescriptiveStats(samples)
    mean = stats.mean()
    std = stats.std(ddof=0)
    median = stats.median()

    # KDE on sorted samples; mode_idx references sorted list
    sorted_s = sorted(samples)
    bandwidth = std * (4.0 / (3.0 * n)) ** 0.2 if std > 0.0 else 1e-6
    kde_vals = _kde_values(sorted_s, bandwidth)
    mode_idx = max(range(n), key=lambda i: kde_vals[i])
    mode_approx = sorted_s[mode_idx]  # index into sorted_s, not original samples

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
    n = len(sorted_samples)
    if bandwidth <= 0.0:
        bandwidth = 1e-6
    inv_denom = 1.0 / (n * bandwidth * math.sqrt(2.0 * math.pi))
    densities = []
    for x in sorted_samples:
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
    if len(grid) < 2:
        raise ValueError("grid must have at least 2 points")
    if len(grid) != len(log_prior_values) or len(grid) != len(log_likelihood_values):
        raise ValueError("grid, log_prior_values, log_likelihood_values must have same length")

    delta = (grid[-1] - grid[0]) / (len(grid) - 1)
    log_vals = [lp + ll for lp, ll in zip(log_prior_values, log_likelihood_values)]
    max_lv = max(log_vals)
    log_integral = max_lv + math.log(
        sum(math.exp(lv - max_lv) for lv in log_vals) * delta
    )
    return log_integral


def bayes_factor(log_ml1: float, log_ml2: float) -> dict:
    log_K = log_ml1 - log_ml2
    try:
        K = math.exp(log_K)
    except OverflowError:
        K = float("inf")

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

    return {"K": K, "log_K": log_K, "evidence": evidence,
            "log_ml1": log_ml1, "log_ml2": log_ml2}


def sequential_update(model, data_stream: list) -> list:
    snapshots = []
    for i, batch in enumerate(data_stream):
        if isinstance(batch, tuple):
            if len(batch) != 2:
                raise ValueError(
                    f"Tuple batch at step {i+1} must have exactly 2 elements, "
                    f"got {len(batch)}."
                )
            model.update(*batch)
        elif isinstance(batch, list):
            model.update(batch)
        else:
            raise TypeError(
                f"Batch at step {i+1} must be a tuple (for BetaBinomial/GammaPoisson) "
                f"or list (for GaussianGaussian), got {type(batch).__name__}."
            )
        snapshots.append({
            "step": i + 1,
            "posterior_mean": model.posterior_mean(),
            "posterior_std": model.posterior_std(),
        })
    return snapshots


def gelman_rubin(chains: list) -> float:
    if len(chains) < 2:
        raise ValueError("Need at least 2 chains for Gelman-Rubin diagnostic")
    n = min(len(c) for c in chains)
    if n < 2:
        raise ValueError("Each chain must have at least 2 samples")

    m = len(chains)
    trimmed = [c[:n] for c in chains]
    chain_means = [sum(c) / n for c in trimmed]
    grand_mean = sum(chain_means) / m

    B = n * sum((cm - grand_mean) ** 2 for cm in chain_means) / (m - 1)
    W = sum(
        sum((x - chain_means[j]) ** 2 for x in trimmed[j]) / (n - 1)
        for j in range(m)
    ) / m

    if W == 0.0:
        return float("inf")

    var_plus = ((n - 1) * W + B) / n
    return math.sqrt(var_plus / W)
