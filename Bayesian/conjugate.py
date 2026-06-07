import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Probability.continuous import NormalDistribution, BetaDistribution, GammaDistribution


class BetaBinomial:
    def __init__(self, alpha_prior: float, beta_prior: float):
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("alpha_prior and beta_prior must be positive.")
        self.alpha = float(alpha_prior)
        self.beta = float(beta_prior)
        self.prior = BetaDistribution(alpha_prior, beta_prior)
        self.posterior = self.prior

    def update(self, n_trials: int, k_success: int) -> None:
        if not isinstance(n_trials, int) or n_trials < 0:
            raise ValueError("n_trials must be a non-negative integer.")
        if not isinstance(k_success, int) or k_success < 0:
            raise ValueError("k_success must be a non-negative integer.")
        if k_success > n_trials:
            raise ValueError(f"k_success ({k_success}) cannot exceed n_trials ({n_trials}).")
        self.alpha += k_success
        self.beta += (n_trials - k_success)
        self.posterior = BetaDistribution(self.alpha, self.beta)

    def posterior_mean(self) -> float:
        return self.posterior.mean()

    def posterior_std(self) -> float:
        return math.sqrt(self.posterior.variance())

    def map_estimate(self) -> float:
        a, b = self.alpha, self.beta
        if a > 1.0 and b > 1.0:
            return (a - 1.0) / (a + b - 2.0)
        if a > 1.0:
            return 1.0
        if b > 1.0:
            return 0.0
        return 0.5

    def credible_interval(self, prob: float = 0.95) -> dict:
        if not (0.0 < prob < 1.0):
            raise ValueError("prob must be strictly between 0 and 1.")
        lower = self.posterior.ppf((1.0 - prob) / 2.0)
        upper = self.posterior.ppf((1.0 + prob) / 2.0)
        return {
            "prob": prob,
            "lower": lower,
            "upper": upper,
            "mean": self.posterior.mean(),
            "std": self.posterior_std(),
        }


class GammaPoisson:
    def __init__(self, alpha_prior: float, beta_prior: float):
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("alpha_prior and beta_prior must be positive.")
        self.alpha = float(alpha_prior)
        self.beta = float(beta_prior)
        self.prior = GammaDistribution(alpha_prior, beta_prior)
        self.posterior = self.prior

    def update(self, k_events: float, t_exposure: float = 1.0) -> None:
        if k_events < 0:
            raise ValueError("k_events must be non-negative.")
        if t_exposure <= 0.0:
            raise ValueError("t_exposure must be positive.")
        self.alpha += k_events
        self.beta += t_exposure
        self.posterior = GammaDistribution(self.alpha, self.beta)

    def posterior_mean(self) -> float:
        return self.posterior.mean()

    def posterior_std(self) -> float:
        return math.sqrt(self.posterior.variance())

    def map_estimate(self) -> float:
        if self.alpha >= 1.0:
            return (self.alpha - 1.0) / self.beta
        return 0.0

    def credible_interval(self, prob: float = 0.95) -> dict:
        if not (0.0 < prob < 1.0):
            raise ValueError("prob must be strictly between 0 and 1.")
        lower = self.posterior.ppf((1.0 - prob) / 2.0)
        upper = self.posterior.ppf((1.0 + prob) / 2.0)
        return {
            "prob": prob,
            "lower": lower,
            "upper": upper,
            "mean": self.posterior.mean(),
            "std": self.posterior_std(),
        }


class GaussianGaussian:
    def __init__(self, prior_mu: float, prior_sigma: float, obs_sigma: float):
        if prior_sigma <= 0.0 or obs_sigma <= 0.0:
            raise ValueError("Standard deviations must be positive.")
        self.prior_mu = float(prior_mu)
        self.prior_var = prior_sigma ** 2
        self.obs_var = obs_sigma ** 2
        self.prior = NormalDistribution(prior_mu, prior_sigma)
        self.posterior = self.prior

    def update(self, data: list) -> None:
        if not isinstance(data, list):
            raise TypeError(f"data must be a list, got {type(data).__name__}.")
        if len(data) == 0:
            raise ValueError("data list must not be empty.")
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

    def posterior_mean(self) -> float:
        return self.posterior.mu

    def posterior_std(self) -> float:
        return self.posterior.sigma

    def map_estimate(self) -> float:
        return self.posterior.mu

    def credible_interval(self, prob: float = 0.95) -> dict:
        if not (0.0 < prob < 1.0):
            raise ValueError("prob must be strictly between 0 and 1.")
        z = NormalDistribution(0.0, 1.0).ppf((1.0 + prob) / 2.0)
        mu = self.posterior.mu
        std = self.posterior.sigma
        return {
            "prob": prob,
            "lower": mu - z * std,
            "upper": mu + z * std,
            "mean": mu,
            "std": std,
        }
