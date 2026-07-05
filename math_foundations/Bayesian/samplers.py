"""
samplers.py
============

Monte Carlo tools for approximate Bayesian inference: grid approximation,
rejection sampling, a random-walk Metropolis-Hastings sampler, and an
autocorrelation-based effective sample size (ESS) estimator for
diagnosing MCMC chain quality.

Example
-------
>>> import random
>>> random.seed(0)
>>> def log_post(theta):
...     return -0.5 * (theta - 2.0) ** 2
>>> result = metropolis_hastings(log_post, init=0.0, n_samples=2000, burn_in=200)
>>> 1.5 < sum(result['samples']) / len(result['samples']) < 2.5
True
"""

from __future__ import annotations

import logging
import math
import random
from typing import Callable, Dict, List, Sequence, Union

Number = Union[int, float]
logger = logging.getLogger(__name__)


def grid_approximation(
    prior_fn: Callable[[float], float],
    likelihood_fn: Callable[[float], float],
    grid: Sequence[Number],
) -> Dict:
    """Discretized (grid) Bayesian posterior approximation.

    Evaluates ``prior(theta) * likelihood(theta)`` at each grid point and
    normalizes via a simple Riemann sum (uniform grid spacing required).

    Parameters
    ----------
    prior_fn, likelihood_fn : Callable[[float], float]
        Prior density and likelihood, each evaluated pointwise.
    grid : Sequence[Number]
        Strictly increasing, uniformly-spaced parameter values; at least
        2 points.

    Returns
    -------
    dict
        Keys: ``grid``, ``posterior`` (normalized), ``log_unnorm``,
        ``normalisation_constant``.

    Raises
    ------
    ValueError
        If ``grid`` has fewer than 2 points, is not strictly increasing,
        is not uniformly spaced, or the posterior is zero everywhere /
        underflows to zero everywhere.
    """
    if len(grid) < 2:
        raise ValueError("grid must contain at least 2 points")

    deltas = [grid[i + 1] - grid[i] for i in range(len(grid) - 1)]
    if min(deltas) <= 0.0:
        raise ValueError(
            "grid must be strictly increasing (found a non-positive step); "
            "a descending or non-monotonic grid silently flips the sign of "
            "the normalization constant."
        )
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
        "grid": list(grid),
        "posterior": posterior,
        "log_unnorm": log_unnorm,
        "normalisation_constant": total,
    }


def rejection_sampling(
    target_fn: Callable[[float], float],
    proposal_sampler: Callable[[], float],
    proposal_fn: Callable[[float], float],
    M: float,
    n_samples: int,
    max_iter: int = 1_000_000,
    bound_check_tol: float = 1e-9,
) -> Dict:
    """Rejection sampling from ``target_fn`` using a proposal distribution scaled by ``M``.

    Requires ``target_fn(x) <= M * proposal_fn(x)`` for all ``x`` in the
    support (the caller's responsibility to ensure); violations are
    detected and logged (see ``bound_check_tol``) rather than silently
    producing a biased sample, since a violated bound means the accepted
    samples no longer follow the target distribution.

    Parameters
    ----------
    target_fn : Callable[[float], float]
        Unnormalized target density.
    proposal_sampler : Callable[[], float]
        Draws one sample from the proposal distribution.
    proposal_fn : Callable[[float], float]
        Proposal density, evaluated pointwise.
    M : float
        Envelope scaling constant; must be positive.
    n_samples : int
        Number of accepted samples to collect; must be positive.
    max_iter : int, optional
        Safety cap on total proposals (accepted + rejected).
    bound_check_tol : float, optional
        Tolerance for detecting ``target_fn(x) > M * proposal_fn(x)``
        (an invalid ``M``); logs a warning (once) rather than raising,
        since a single warning is enough to alert the caller without
        aborting a long-running sampling job.

    Returns
    -------
    dict
        Keys: ``samples``, ``n_accepted``, ``n_rejected``, ``acceptance_rate``.

    Raises
    ------
    ValueError
        If ``M``, ``n_samples``, or ``max_iter`` is not positive.
    RuntimeError
        If ``max_iter`` proposals are exhausted before collecting
        ``n_samples`` accepted draws, or ``target_fn`` raises.
    """
    if M <= 0.0:
        raise ValueError("M must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    samples = []
    accepted = 0
    rejected = 0
    bound_violation_warned = False

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

        envelope = M * p_x
        if t_x > envelope + bound_check_tol and not bound_violation_warned:
            logger.warning(
                "target_fn(x=%s)=%s exceeds M*proposal_fn(x)=%s: the envelope "
                "bound is violated, so accepted samples will NOT follow the "
                "target distribution. Increase M.",
                x, t_x, envelope,
            )
            bound_violation_warned = True

        accept_prob = min(1.0, t_x / envelope)
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


def metropolis_hastings(
    log_posterior_fn: Callable[[float], float],
    init: float,
    n_samples: int,
    step_size: float = 0.1,
    burn_in: int = 500,
) -> Dict:
    """Random-walk Metropolis-Hastings sampler with a Gaussian proposal.

    Parameters
    ----------
    log_posterior_fn : Callable[[float], float]
        Unnormalized log-posterior density.
    init : float
        Starting value.
    n_samples : int
        Number of post-burn-in samples to keep; must be positive.
    step_size : float, optional
        Standard deviation of the Gaussian random-walk proposal; must be positive.
    burn_in : int, optional
        Number of initial iterations to discard; must be non-negative.

    Returns
    -------
    dict
        Keys: ``samples``, ``acceptance_rate`` (post-burn-in only),
        ``n_samples``, ``burn_in``, ``step_size``, ``init``.

    Raises
    ------
    ValueError
        If ``n_samples``/``step_size``/``burn_in`` are invalid, or the
        initial log-posterior is ``-inf``/NaN (an invalid starting point).
    """
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

    if isinstance(current_log_p, float) and math.isnan(current_log_p):
        raise ValueError(f"log_posterior_fn(init={init}) is NaN; choose a valid starting point")
    if current_log_p == float("-inf"):
        raise ValueError(
            f"log_posterior_fn(init={init}) is -inf (zero posterior density); "
            "choose a starting point with positive posterior density"
        )

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


def effective_sample_size(samples: Sequence[Number], max_lag: int = 1000) -> float:
    """Autocorrelation-based effective sample size (ESS) of an MCMC chain.

    Sums the lag-k autocorrelation ``rho_k`` until it drops below 0.05 in
    magnitude (a common simple truncation heuristic; not as rigorous as
    Geyer's initial monotone sequence estimator, but standard for a
    from-scratch implementation).

    Parameters
    ----------
    samples : Sequence[Number]
        The chain.
    max_lag : int, optional
        Hard cap on the number of lags examined, so pathological chains
        (autocorrelation decaying very slowly) don't force an O(n^2)
        blow-up for very long chains.

    Returns
    -------
    float
        Estimated ESS, clamped to ``[1, len(samples)]``.
    """
    n = len(samples)
    if n < 4:
        return float(n)

    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    if var == 0.0:
        return 1.0

    rho_sum = 0.0
    lag_cap = min(n - 1, max_lag)
    for lag in range(1, lag_cap + 1):
        cov = sum(
            (samples[i] - mean) * (samples[i + lag] - mean)
            for i in range(n - lag)
        ) / n
        rho = cov / var
        if abs(rho) < 0.05:  # truncate on |rho|, not rho, to handle anti-correlation
            break
        rho_sum += rho

    denom = 1.0 + 2.0 * rho_sum
    if denom <= 0.0:
        # Extreme anti-correlation pushed the denominator non-positive;
        # this is a degenerate/pathological case (e.g. a chain that
        # perfectly alternates), so fall back to the most conservative
        # estimate rather than dividing by zero or returning a negative ESS.
        return 1.0

    ess = n / denom
    return min(float(n), max(1.0, ess))  # ESS cannot exceed n
