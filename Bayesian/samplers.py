import math
import random


def grid_approximation(prior_fn, likelihood_fn, grid: list) -> dict:
    """
    Approximate posterior on a uniform grid using the trapezoidal rule.
    """
    if len(grid) < 2:
        raise ValueError("grid must contain at least 2 points")

    # Validate uniform spacing (optional but helpful)
    deltas = [grid[i+1] - grid[i] for i in range(len(grid)-1)]
    if max(deltas) - min(deltas) > 1e-12:
        raise ValueError("grid must be uniformly spaced")

    log_unnorm = []
    for theta in grid:
        lp = prior_fn(theta)
        ll = likelihood_fn(theta)
        if lp <= 0.0 or ll <= 0.0:
            log_unnorm.append(float("-inf"))
        else:
            log_unnorm.append(math.log(lp) + math.log(ll))

    finite_logs = [lv for lv in log_unnorm if lv != float("-inf")]
    if not finite_logs:
        raise ValueError("Posterior is zero everywhere on the grid")

    max_log = max(finite_logs)
    delta = deltas[0]   # already checked uniformity
    unnorm = [math.exp(lv - max_log) if lv != float("-inf") else 0.0
            for lv in log_unnorm]
    total = sum(unnorm) * delta
    if total == 0.0:
        raise ValueError("Posterior normalisation constant is zero (numerical underflow)")

    posterior = [u / total for u in unnorm]
    return {
        "grid": grid,
        "posterior": posterior,
        "log_unnorm": log_unnorm,
        "normalisation_constant": total,
    }


def rejection_sampling(
    target_fn,
    proposal_sampler,
    proposal_fn,
    M: float,
    n_samples: int,
    max_iter: int = 1_000_000,
) -> dict:
    """
    Rejection sampling from an unnormalised target density.

    The target must satisfy target(x) <= M * proposal_fn(x) for all x.
    """
    if M <= 0.0:
        raise ValueError("M must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    samples = []
    accepted = 0
    rejected = 0

    while accepted < n_samples:
        if accepted + rejected >= max_iter:
            raise RuntimeError(
                f"Rejection sampling exceeded {max_iter} iterations. "
                f"Only {accepted}/{n_samples} samples collected. "
                f"Consider increasing M or max_iter."
            )
        x = proposal_sampler()
        p_x = proposal_fn(x)
        if p_x <= 0.0:
            rejected += 1
            continue

        # acceptance probability (capped at 1.0 for safety if M is slightly too small)
        accept_prob = target_fn(x) / (M * p_x)
        if accept_prob > 1.0:
            # Warn? – still accept with prob 1, but indicate M may be too small.
            accept_prob = 1.0

        u = random.random()
        if u < accept_prob:
            samples.append(x)
            accepted += 1
        else:
            rejected += 1

    return {
        "samples": samples,
        "n_accepted": accepted,
        "n_rejected": rejected,
        "acceptance_rate": accepted / (accepted + rejected),
    }


def metropolis_hastings(
    log_posterior_fn,
    init: float,
    n_samples: int,
    step_size: float = 0.1,
    burn_in: int = 500,
) -> dict:
    """
    Metropolis–Hastings random‑walk sampler with Gaussian proposals.

    log_posterior_fn(x) returns the log of the unnormalised posterior at x.
    Returns n_samples after burn‑in.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")

    samples = []
    current = init

    # initial log posterior – allow -inf (non‑positive density)
    try:
        current_log_p = log_posterior_fn(current)
    except Exception as e:
        raise ValueError(f"log_posterior_fn failed at initial value {init}: {e}")

    n_total = n_samples + burn_in
    n_accepted = 0
    for i in range(n_total):
        # Gaussian random walk proposal
        proposed = current + random.gauss(0.0, step_size)

        try:
            proposed_log_p = log_posterior_fn(proposed)
        except Exception:
            proposed_log_p = float("-inf")

        # Metropolis acceptance ratio (log scale)
        log_alpha = proposed_log_p - current_log_p
        u = random.random()
        log_u = math.log(u) if u > 0.0 else float("-inf")

        if log_u < log_alpha:
            current = proposed
            current_log_p = proposed_log_p
            if i >= burn_in:
                n_accepted += 1

        # store sample after burn‑in (always store current state, regardless of acceptance)
        if i >= burn_in:
            samples.append(current)

    acceptance_rate = n_accepted / n_samples if n_samples > 0 else 0.0

    return {
        "samples": samples,
        "acceptance_rate": acceptance_rate,
        "n_samples": n_samples,
        "burn_in": burn_in,
        "step_size": step_size,
        "init": init,
    }


def effective_sample_size(samples: list) -> float:
    """
    Estimate the effective sample size (ESS) using the sample autocorrelation.

    ESS = n / (1 + 2 * sum_{lag=1}^{∞} ρ_lag)
    The sum is truncated when ρ_lag falls below 0.05 (common heuristic).
    """
    n = len(samples)
    if n < 4:
        return float(n)

    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    if var == 0.0:
        return 1.0

    rho_sum = 0.0
    for lag in range(1, n):
        cov = sum((samples[i] - mean) * (samples[i + lag] - mean)
                for i in range(n - lag)) / n
        rho = cov / var
        if rho < 0.05:   # negligible correlation
            break
        rho_sum += rho

    ess = n / (1.0 + 2.0 * rho_sum)
    return max(1.0, ess)
