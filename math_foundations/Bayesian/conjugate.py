"""
conjugate.py
============

Closed-form Bayesian conjugate-prior/posterior update models: Beta-Binomial
(unknown success probability), Gamma-Poisson (unknown rate), and
Gaussian-Gaussian (unknown mean, known variance).

Every model shares the same interface -- ``update(...)``, ``posterior_mean()``,
``posterior_std()``, ``map_estimate()``, ``credible_interval(prob)`` -- via
the common :class:`ConjugateModel` base class, which implements
``posterior_mean``/``posterior_std``/``credible_interval`` once in terms of
the concrete model's ``self.posterior`` distribution object (any of
``NormalDistribution``/``BetaDistribution``/``GammaDistribution`` from
:mod:`continuous`). Only ``update`` and ``map_estimate`` are genuinely
model-specific and need overriding.

Example
-------
>>> model = BetaBinomial(alpha_prior=1.0, beta_prior=1.0)
>>> model.update(n_trials=10, k_success=7)
>>> round(model.posterior_mean(), 4)
0.6667
"""

from __future__ import annotations

import os
import sys
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Union

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from Probability.continuous import (  # type: ignore
    BetaDistribution,
    GammaDistribution,
    NormalDistribution,
)

Number = Union[int, float]


def _check_positive(value: float, name: str) -> None:
    """Validate that ``value`` is a positive, finite, non-NaN, non-bool real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    if math.isinf(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


class ConjugateModel(ABC):
    """Common interface for closed-form conjugate Bayesian update models.

    Subclasses must set ``self.posterior`` to a distribution object
    exposing ``mean()``, ``variance()``, and ``ppf(p)`` (any of this
    package's continuous distributions satisfies that), and must
    implement :meth:`update` and :meth:`map_estimate`.
    """

    posterior: Union[BetaDistribution, GammaDistribution, NormalDistribution]

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the posterior in light of new data. Model-specific."""
        raise NotImplementedError

    @abstractmethod
    def map_estimate(self) -> float:
        """Return the maximum a posteriori (mode) point estimate. Model-specific."""
        raise NotImplementedError

    def posterior_mean(self) -> float:
        """Posterior mean."""
        return self.posterior.mean()

    def posterior_std(self) -> float:
        """Posterior standard deviation."""
        return math.sqrt(self.posterior.variance())

    def credible_interval(self, prob: float = 0.95) -> Dict[str, float]:
        """Equal-tailed posterior credible interval.

        Parameters
        ----------
        prob : float, optional
            Credible mass, in ``(0, 1)``. Default 0.95 gives a 95% interval.

        Returns
        -------
        dict
            Keys: ``prob``, ``lower``, ``upper``, ``mean``, ``std``.

        Raises
        ------
        ValueError
            If ``prob`` is not strictly between 0 and 1.
        """
        if not (0.0 < prob < 1.0):
            raise ValueError("prob must be strictly between 0 and 1.")
        lower = self.posterior.ppf((1.0 - prob) / 2.0)
        upper = self.posterior.ppf((1.0 + prob) / 2.0)
        return {
            "prob": prob,
            "lower": lower,
            "upper": upper,
            "mean": self.posterior_mean(),
            "std": self.posterior_std(),
        }


class BetaBinomial(ConjugateModel):
    """Beta prior / Binomial likelihood conjugate model for an unknown success probability.

    Parameters
    ----------
    alpha_prior, beta_prior : float
        Beta distribution shape parameters for the prior; must be positive.
    """

    def __init__(self, alpha_prior: float, beta_prior: float):
        _check_positive(alpha_prior, "alpha_prior")
        _check_positive(beta_prior, "beta_prior")
        self.alpha = float(alpha_prior)
        self.beta = float(beta_prior)
        self.prior = BetaDistribution(alpha_prior, beta_prior)
        self.posterior = self.prior

    def __repr__(self) -> str:
        return f"BetaBinomial(alpha={self.alpha}, beta={self.beta})"

    def update(self, n_trials: int, k_success: int) -> None:
        """Fold in ``k_success`` successes out of ``n_trials`` trials.

        Raises
        ------
        TypeError
            If ``n_trials``/``k_success`` are not ``int`` (bools excluded).
        ValueError
            If either is negative, or ``k_success > n_trials``.
        """
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 0:
            raise ValueError("n_trials must be a non-negative integer.")
        if isinstance(k_success, bool) or not isinstance(k_success, int) or k_success < 0:
            raise ValueError("k_success must be a non-negative integer.")
        if k_success > n_trials:
            raise ValueError(f"k_success ({k_success}) cannot exceed n_trials ({n_trials}).")
        self.alpha += k_success
        self.beta += (n_trials - k_success)
        self.posterior = BetaDistribution(self.alpha, self.beta)

    def map_estimate(self) -> float:
        """Posterior mode.

        Raises
        ------
        ValueError
            If ``alpha <= 1`` and ``beta <= 1``: the density is either
            uniform (``alpha == beta == 1``, every point equally likely)
            or bimodal with divergent peaks at both 0 and 1, so no single
            finite mode exists. Inspect ``posterior.pdf`` directly in
            that case.
        """
        a, b = self.alpha, self.beta
        if a > 1.0 and b > 1.0:
            return (a - 1.0) / (a + b - 2.0)
        if a > 1.0:
            return 1.0
        if b > 1.0:
            return 0.0
        raise ValueError(
            f"No unique mode exists for Beta(alpha={a}, beta={b}): the density is "
            "uniform (if alpha == beta == 1) or bimodal with divergent peaks at both "
            "0 and 1 (if alpha < 1 or beta < 1). Inspect posterior.pdf(0) / "
            "posterior.pdf(1) directly instead of requesting a single MAP estimate."
        )


class GammaPoisson(ConjugateModel):
    """Gamma prior / Poisson likelihood conjugate model for an unknown rate.

    Parameters
    ----------
    alpha_prior : float
        Gamma shape parameter for the prior; must be positive.
    beta_prior : float
        Gamma rate parameter for the prior; must be positive.
    """

    def __init__(self, alpha_prior: float, beta_prior: float):
        _check_positive(alpha_prior, "alpha_prior")
        _check_positive(beta_prior, "beta_prior")
        self.alpha = float(alpha_prior)
        self.beta = float(beta_prior)
        self.prior = GammaDistribution(alpha_prior, beta_prior)
        self.posterior = self.prior

    def __repr__(self) -> str:
        return f"GammaPoisson(alpha={self.alpha}, beta={self.beta})"

    def update(self, k_events: float, t_exposure: float = 1.0) -> None:
        """Fold in ``k_events`` observed over ``t_exposure`` units of exposure.

        Raises
        ------
        TypeError
            If ``k_events``/``t_exposure`` are not real numbers.
        ValueError
            If ``k_events`` is negative or ``t_exposure`` is non-positive.
        """
        if isinstance(k_events, bool) or not isinstance(k_events, (int, float)):
            raise TypeError(f"k_events must be a real number, got {type(k_events).__name__}")
        if isinstance(t_exposure, bool) or not isinstance(t_exposure, (int, float)):
            raise TypeError(f"t_exposure must be a real number, got {type(t_exposure).__name__}")
        if k_events < 0:
            raise ValueError("k_events must be non-negative.")
        if t_exposure <= 0.0:
            raise ValueError("t_exposure must be positive.")
        self.alpha += k_events
        self.beta += t_exposure
        self.posterior = GammaDistribution(self.alpha, self.beta)

    def map_estimate(self) -> float:
        """Posterior mode: ``(alpha - 1) / beta`` for ``alpha >= 1``, else ``0.0`` (boundary)."""
        if self.alpha >= 1.0:
            return (self.alpha - 1.0) / self.beta
        return 0.0


class GaussianGaussian(ConjugateModel):
    """Normal prior / Normal likelihood conjugate model for an unknown mean, known variance.

    Parameters
    ----------
    prior_mu : float
        Prior mean.
    prior_sigma : float
        Prior standard deviation; must be positive.
    obs_sigma : float
        *Known* per-observation standard deviation; must be positive.
    """

    def __init__(self, prior_mu: float, prior_sigma: float, obs_sigma: float):
        if isinstance(prior_mu, bool) or not isinstance(prior_mu, (int, float)) or math.isnan(prior_mu):
            raise ValueError("prior_mu must be a finite real number")
        _check_positive(prior_sigma, "prior_sigma")
        _check_positive(obs_sigma, "obs_sigma")
        self.prior_mu = float(prior_mu)
        self.prior_var = prior_sigma ** 2
        self.obs_var = obs_sigma ** 2
        self.prior = NormalDistribution(prior_mu, prior_sigma)
        self.posterior = self.prior

    def __repr__(self) -> str:
        return f"GaussianGaussian(mu={self.prior_mu}, prior_var={self.prior_var}, obs_var={self.obs_var})"

    def update(self, data: Sequence[Number]) -> None:
        """Fold in a batch of observations (precision-weighted conjugate update).

        Parameters
        ----------
        data : Sequence[Number]
            Non-empty sequence of observations (list or tuple; not a raw string).

        Raises
        ------
        TypeError
            If ``data`` is not a list/tuple, or contains non-numeric elements.
        ValueError
            If ``data`` is empty, or contains NaN/Inf.
        """
        if isinstance(data, str) or not isinstance(data, (list, tuple)):
            raise TypeError(f"data must be a list or tuple, got {type(data).__name__}.")
        if len(data) == 0:
            raise ValueError("data must not be empty.")
        for i, x in enumerate(data):
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise TypeError(f"data[{i}] must be numeric, got {type(x).__name__}.")
            if math.isnan(x) or math.isinf(x):
                raise ValueError(f"data[{i}] must be finite, got {x}.")

        n = len(data)
        x_bar = sum(data) / n
        prior_prec = 1.0 / self.prior_var
        obs_prec = n / self.obs_var
        post_prec = prior_prec + obs_prec
        post_var = 1.0 / post_prec
        post_mu = (prior_prec * self.prior_mu + obs_prec * x_bar) / post_prec
        self.prior_mu = post_mu
        self.prior_var = post_var
        self.posterior = NormalDistribution(post_mu, math.sqrt(post_var))

    def map_estimate(self) -> float:
        """Posterior mode: for a Normal distribution this equals the mean."""
        return self.posterior.mu
