import math
import random


def grid_approximation(prior_fn, likelihood_fn, grid: list) -> dict:
    if len(grid) < 2:
        raise ValueError("grid must contain at least 2 points")

    deltas = [grid[i + 1] - grid[i] for i in range(len(grid) - 1)]
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

    finite_logs = [lv for lv in log_unnorm if math.isfinite(lv)]
    if not finite_logs:
        raise ValueError("Posterior is zero everywhere on the grid")

    max_log = max(finite_logs)
    delta = deltas[0]
    unnorm = [math.exp(lv - max_log) if math.isfinite(lv) else 0.0
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


def rejection_sampling(target_fn,proposal_sampler,proposal_fn,M: float,n_samples: int,max_iter: int = 1_000_000,) -> dict:
    if M <= 0.0:
        raise ValueError("M must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

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

        try:
            t_x = target_fn(x)
        except Exception as e:
            raise RuntimeError(f"target_fn raised an exception at x={x}: {e}")

        accept_prob = min(1.0, t_x / (M * p_x))
        if random.random() < accept_prob:
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


def metropolis_hastings(log_posterior_fn,init: float,n_samples: int,step_size: float = 0.1,burn_in: int = 500,) -> dict:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")

    try:
        current_log_p = log_posterior_fn(init)
    except Exception as e:
        raise ValueError(f"log_posterior_fn failed at initial value {init}: {e}")

    current = init
    samples = []
    n_accepted = 0

    for i in range(n_samples + burn_in):
        proposed = current + random.gauss(0.0, step_size)

        try:
            proposed_log_p = log_posterior_fn(proposed)
        except Exception:
            proposed_log_p = float("-inf")

        log_alpha = proposed_log_p - current_log_p
        u = random.random()
        log_u = math.log(u) if u > 0.0 else float("-inf")

        if log_u < log_alpha:
            current = proposed
            current_log_p = proposed_log_p
            if i >= burn_in:
                n_accepted += 1

        if i >= burn_in:
            samples.append(current)

    return {
        "samples": samples,
        "acceptance_rate": n_accepted / n_samples,
        "n_samples": n_samples,
        "burn_in": burn_in,
        "step_size": step_size,
        "init": init,
    }


def effective_sample_size(samples: list) -> float:
    n = len(samples)
    if n < 4:
        return float(n)

    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    if var == 0.0:
        return 1.0

    rho_sum = 0.0
    for lag in range(1, n):
        cov = sum(
            (samples[i] - mean) * (samples[i + lag] - mean)
            for i in range(n - lag)
        ) / n
        rho = cov / var
        if abs(rho) < 0.05:  # truncate on |rho|, not rho, to handle anti-correlation
            break
        rho_sum += rho

    ess = n / (1.0 + 2.0 * rho_sum)
    return min(float(n), max(1.0, ess))  # ESS cannot exceed n
