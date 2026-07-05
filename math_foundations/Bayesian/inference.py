"""
inference.py
=============

Sample-based and grid-based Bayesian posterior summary utilities: MAP
estimation, equal-tailed credible intervals, highest-density intervals
(HDI), full posterior summaries (with a KDE-based mode estimate), log
marginal likelihood via numerical quadrature, Bayes factors, sequential
(streaming) conjugate updates, and the Gelman-Rubin MCMC convergence
diagnostic.

Example
-------
>>> samples = [0.61, 0.58, 0.65, 0.60, 0.63, 0.59, 0.62]
>>> ci = credible_interval(samples, prob=0.90)
>>> round(ci['lower'], 2), round(ci['upper'], 2)
(0.58, 0.65)
"""

from __future__ import annotations

import os
import sys
import logging
import math
from typing import Callable, Dict, List, Sequence, Tuple, Union

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from Probability.descriptive_stats import DescriptiveStats  # type: ignore

Number = Union[int, float]
logger = logging.getLogger(__name__)


def _validate_samples(samples: Sequence[Number], name: str = "samples") -> None:
    """Validate a non-empty sequence of finite, non-NaN, non-bool numbers.

    Raises
    ------
    ValueError
        If ``samples`` is empty or contains NaN.
    TypeError
        If any element is non-numeric.
    """
    if not samples:
        raise ValueError(f"{name} list is empty")
    for i, x in enumerate(samples):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(f"{name}[{i}] must be numeric, got {type(x).__name__}")
        if isinstance(x, float) and math.isnan(x):
            raise ValueError(f"{name}[{i}] is NaN; sorting-based statistics are undefined for NaN")


def map_estimate(grid: Sequence[Number], posterior_values: Sequence[Number]) -> Number:
    """Grid-search MAP estimate: the grid point with the highest (unnormalized) posterior value.

    Parameters
    ----------
    grid : Sequence[Number]
        Parameter values evaluated.
    posterior_values : Sequence[Number]
        (Possibly unnormalized) posterior density/mass at each grid point;
        need not sum/integrate to 1.

    Returns
    -------
    Number
        The grid point achieving the maximum. On ties, the first
        occurrence (lowest index) is returned.

    Raises
    ------
    ValueError
        If ``grid`` is empty, lengths mismatch, or ``posterior_values``
        contains negative values or NaN.
    """
    if len(grid) == 0:
        raise ValueError("grid must be non-empty")
    if len(grid) != len(posterior_values):
        raise ValueError("grid and posterior_values must have the same length")
    for i, p in enumerate(posterior_values):
        if isinstance(p, float) and math.isnan(p):
            raise ValueError(f"posterior_values[{i}] is NaN")
        if p < 0:
            raise ValueError(f"posterior_values[{i}] is negative ({p}); posterior values must be non-negative")
    best_idx = max(range(len(posterior_values)), key=lambda i: posterior_values[i])
    return grid[best_idx]


def credible_interval(samples: Sequence[Number], prob: float = 0.95) -> Dict[str, float]:
    """Equal-tailed posterior credible interval from samples (empirical percentiles).

    Parameters
    ----------
    samples : Sequence[Number]
        Posterior samples (e.g. from MCMC).
    prob : float, optional
        Credible mass, in ``(0, 1)``.

    Returns
    -------
    dict
        Keys: ``prob``, ``lower``, ``upper``.

    Raises
    ------
    ValueError
        If ``samples`` is empty/contains NaN, or ``prob`` is not in ``(0, 1)``.
    """
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must be strictly between 0 and 1")
    stats = DescriptiveStats(samples)  # validates non-empty, numeric, no NaN
    alpha = (1.0 - prob) / 2.0
    return {
        "prob": prob,
        "lower": stats.percentile(100.0 * alpha),
        "upper": stats.percentile(100.0 * (1.0 - alpha)),
    }


def hdi(samples: Sequence[Number], prob: float = 0.95) -> Dict[str, float]:
    """Highest Density Interval: the shortest interval containing ``prob`` mass.

    Finds the narrowest window covering ``ceil(prob * n)`` sorted sample
    points via a linear scan (Chen & Shao, 1999).

    Parameters
    ----------
    samples : Sequence[Number]
        Posterior samples; needs at least 2 points.
    prob : float, optional
        Credible mass, in ``(0, 1)``.

    Returns
    -------
    dict
        Keys: ``prob``, ``lower``, ``upper``, ``width``.

    Raises
    ------
    ValueError
        If fewer than 2 samples, samples contain NaN, or ``prob`` is not in ``(0, 1)``.
    """
    _validate_samples(samples)
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


def _kde_mode(sorted_samples: List[Number], bandwidth: float, grid_size: int = 500) -> float:
    """Approximate the density mode via Gaussian KDE evaluated on a fixed-size grid.

    Evaluates the KDE at ``grid_size`` evenly-spaced points spanning the
    data range rather than at every one of the ``n`` samples: this is
    O(n * grid_size) instead of the O(n^2) cost of evaluating at every
    sample point, which matters for large MCMC chains (n in the tens of
    thousands) and also yields a mode estimate that isn't artificially
    restricted to landing exactly on an observed sample.

    Parameters
    ----------
    sorted_samples : list of Number
        Samples, already sorted ascending.
    bandwidth : float
        KDE bandwidth (must be positive; caller is responsible for a
        sane value, e.g. via Silverman's rule).
    grid_size : int, optional
        Number of evaluation points. Capped at ``len(sorted_samples)``
        for very small sample sets (no point using more grid points than
        data).

    Returns
    -------
    float
        The grid location of maximum estimated density.
    """
    n = len(sorted_samples)
    lo, hi = sorted_samples[0], sorted_samples[-1]
    if lo == hi:
        return lo

    m = max(2, min(grid_size, n))
    step = (hi - lo) / (m - 1)
    eval_points = [lo + i * step for i in range(m)]

    inv_denom = 1.0 / (n * bandwidth * math.sqrt(2.0 * math.pi))
    best_density = -1.0
    best_x = eval_points[0]
    for x in eval_points:
        density = sum(math.exp(-0.5 * ((x - xi) / bandwidth) ** 2) for xi in sorted_samples) * inv_denom
        if density > best_density:
            best_density = density
            best_x = x
    return best_x


def posterior_summary(samples: Sequence[Number], ci_prob: float = 0.95, kde_grid_size: int = 500) -> Dict:
    """A one-stop summary of posterior samples: mean, std, median, KDE mode, CI, and HDI.

    Parameters
    ----------
    samples : Sequence[Number]
        Posterior samples; needs at least 2 points.
    ci_prob : float, optional
        Credible mass used for both the equal-tailed CI and the HDI.
    kde_grid_size : int, optional
        Grid resolution for the KDE-based mode estimate; see :func:`_kde_mode`.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``median``, ``mode_approx``,
        ``ci_<pct>``, ``hdi_<pct>``, ``n_samples``.

    Raises
    ------
    ValueError
        If fewer than 2 samples, samples contain NaN, or ``ci_prob`` is not in ``(0, 1)``.
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples for summary statistics")
    if not (0.0 < ci_prob < 1.0):
        raise ValueError("ci_prob must be strictly between 0 and 1")

    n = len(samples)
    stats = DescriptiveStats(samples)  # validates numeric, no NaN
    mean = stats.mean()
    std = stats.std(ddof=0)
    median = stats.median()

    sorted_s = sorted(samples)
    bandwidth = std * (4.0 / (3.0 * n)) ** 0.2 if std > 0.0 else 1e-6
    mode_approx = _kde_mode(sorted_s, bandwidth, grid_size=kde_grid_size)

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


def log_marginal_likelihood(
    grid: Sequence[Number],
    log_prior_values: Sequence[Number],
    log_likelihood_values: Sequence[Number],
) -> float:
    """Log marginal likelihood (model evidence) via trapezoidal quadrature in log-space.

    Computes ``log integral[ prior(theta) * likelihood(theta) dtheta ]``
    using the trapezoidal rule (supports non-uniformly spaced grids,
    unlike a naive fixed-delta Riemann sum), evaluated stably via the
    log-sum-exp trick to avoid overflow/underflow.

    Parameters
    ----------
    grid : Sequence[Number]
        Strictly increasing parameter grid; at least 2 points.
    log_prior_values, log_likelihood_values : Sequence[Number]
        Log-prior and log-likelihood evaluated at each grid point.

    Returns
    -------
    float
        The log marginal likelihood.

    Raises
    ------
    ValueError
        If lengths mismatch, fewer than 2 grid points, the grid is not
        strictly increasing, or any input contains NaN.
    """
    if len(grid) < 2:
        raise ValueError("grid must have at least 2 points")
    if len(grid) != len(log_prior_values) or len(grid) != len(log_likelihood_values):
        raise ValueError("grid, log_prior_values, log_likelihood_values must have same length")
    for i in range(len(grid) - 1):
        if grid[i + 1] <= grid[i]:
            raise ValueError(
                f"grid must be strictly increasing; grid[{i}]={grid[i]} >= grid[{i+1}]={grid[i+1]}"
            )

    n = len(grid)
    log_vals = [lp + ll for lp, ll in zip(log_prior_values, log_likelihood_values)]
    for i, lv in enumerate(log_vals):
        if isinstance(lv, float) and math.isnan(lv):
            raise ValueError(f"log_prior_values[{i}] + log_likelihood_values[{i}] is NaN")

    # Trapezoidal weights: each interior point gets half the sum of its two
    # neighboring intervals; endpoints get half of their single adjacent
    # interval. This generalizes correctly to non-uniform grids (a uniform
    # grid reduces to the standard trapezoidal rule with constant weight
    # `delta` for interior points and `delta/2` at the ends).
    weights = [0.0] * n
    weights[0] = (grid[1] - grid[0]) / 2.0
    weights[-1] = (grid[-1] - grid[-2]) / 2.0
    for i in range(1, n - 1):
        weights[i] = (grid[i + 1] - grid[i - 1]) / 2.0

    # log(f_i * w_i) = log_vals[i] + log(w_i), then log-sum-exp for the
    # overall log integral -- numerically stable even when log_vals span
    # a huge dynamic range.
    weighted_log_vals = [lv + math.log(w) for lv, w in zip(log_vals, weights)]
    max_lv = max(weighted_log_vals)
    log_integral = max_lv + math.log(sum(math.exp(wv - max_lv) for wv in weighted_log_vals))
    return log_integral


def bayes_factor(log_ml1: float, log_ml2: float) -> Dict:
    """Bayes factor K = ML1/ML2 with a qualitative interpretation (Jeffreys' 1961 scale).

    Parameters
    ----------
    log_ml1, log_ml2 : float
        Log marginal likelihoods of models 1 and 2 (e.g. from
        :func:`log_marginal_likelihood`).

    Returns
    -------
    dict
        Keys: ``K``, ``log_K``, ``evidence``, ``log_ml1``, ``log_ml2``.

    Raises
    ------
    ValueError
        If either input is NaN.
    """
    if math.isnan(log_ml1) or math.isnan(log_ml2):
        raise ValueError("log_ml1 and log_ml2 must not be NaN")

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


def sequential_update(model, data_stream: Sequence) -> List[Dict]:
    """Apply a stream of data batches to a conjugate model, snapshotting the posterior after each.

    Parameters
    ----------
    model
        Any object with ``update(...)``, ``posterior_mean()``, and
        ``posterior_std()`` methods (e.g. a :class:`conjugate.ConjugateModel`).
    data_stream : Sequence
        Each element is either a 2-tuple ``(arg1, arg2)`` (unpacked as
        positional args, e.g. for ``BetaBinomial``/``GammaPoisson``) or a
        list (passed as a single argument, e.g. for ``GaussianGaussian``).

    Returns
    -------
    list of dict
        One entry per batch: ``{"step", "posterior_mean", "posterior_std"}``.

    Raises
    ------
    TypeError
        If ``model`` lacks the required methods, or a batch is neither a
        2-tuple nor a list.
    ValueError
        If a tuple batch doesn't have exactly 2 elements.
    """
    for method in ("update", "posterior_mean", "posterior_std"):
        if not callable(getattr(model, method, None)):
            raise TypeError(f"model must have a callable '{method}' method")

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


def gelman_rubin(chains: Sequence[Sequence[Number]]) -> float:
    """Gelman-Rubin R-hat convergence diagnostic across multiple MCMC chains.

    Parameters
    ----------
    chains : Sequence[Sequence[Number]]
        At least 2 chains, each with at least 2 samples. Chains of
        differing length are truncated to the shortest chain's length.

    Returns
    -------
    float
        R-hat >= 1.0. Values close to 1.0 (conventionally < 1.1 or
        < 1.01 depending on the field) indicate convergence. Returns
        exactly ``1.0`` if every chain is a constant with an identical
        value (trivial, perfect agreement), and ``inf`` if within-chain
        variance is zero but chains disagree (genuine non-convergence).

    Raises
    ------
    ValueError
        If fewer than 2 chains, any chain has fewer than 2 samples after
        truncation, or any value is NaN.
    """
    if len(chains) < 2:
        raise ValueError("Need at least 2 chains for Gelman-Rubin diagnostic")
    for c in chains:
        _validate_samples(c, "chain")
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
        # Degenerate case: every chain has zero internal variance. If the
        # chains also agree with each other (B == 0 too), this is trivial
        # perfect convergence (e.g. all chains stuck at the same constant),
        # not a divergence signal -- R-hat should read 1.0, not infinity.
        return 1.0 if B == 0.0 else float("inf")

    var_plus = ((n - 1) * W + B) / n
    return math.sqrt(var_plus / W)
