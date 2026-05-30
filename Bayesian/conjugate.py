import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Probability.continuous import NormalDistribution, BetaDistribution, GammaDistribution


class BetaBinomial:
    def __init__(self, alpha_prior, beta_prior):
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.prior = BetaDistribution(alpha_prior, beta_prior)
        self.posterior = self.prior          

    def update(self, n_trials: int, k_success: int):
        self.alpha += k_success
        self.beta += (n_trials - k_success)
        self.posterior = BetaDistribution(self.alpha, self.beta)

    def posterior_mean(self):
        return self.posterior.mean()

    def posterior_std(self):
        return math.sqrt(self.posterior.variance())

    def map_estimate(self):
        """Beta MAP: (α-1)/(α+β-2) for α,β > 1, otherwise boundary."""
        a = self.alpha
        b = self.beta
        if a > 1 and b > 1:
            return (a - 1) / (a + b - 2)
        elif a > 1:          # b <= 1
            return 1.0
        elif b > 1:          # a <= 1
            return 0.0
        else:               # both <= 1, uniform-like, any point; return 0.5
            return 0.5

    def credible_interval(self, prob=0.95):
        lower = self.posterior.ppf((1.0 - prob) / 2.0)
        upper = self.posterior.ppf((1.0 + prob) / 2.0)
        return {
            "prob": prob,
            "lower": lower,
            "upper": upper,
            "mean": self.posterior.mean(),
            "std": self.posterior_std()
        }


class GammaPoisson:
    def __init__(self, alpha_prior, beta_prior):
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.prior = GammaDistribution(alpha_prior, beta_prior)
        self.posterior = self.prior         

    def update(self, k_events, t_exposure=1.0):
        self.alpha += k_events
        self.beta += t_exposure
        self.posterior = GammaDistribution(self.alpha, self.beta)

    def posterior_mean(self):
        return self.posterior.mean()

    def posterior_std(self):
        return math.sqrt(self.posterior.variance())

    def map_estimate(self):
        """Gamma MAP: (α-1)/β for α >= 1, else 0."""
        if self.alpha >= 1:
            return (self.alpha - 1) / self.beta
        else:
            return 0.0

    def credible_interval(self, prob=0.95):
        lower = self.posterior.ppf((1.0 - prob) / 2.0)
        upper = self.posterior.ppf((1.0 + prob) / 2.0)
        return {
            "prob": prob,
            "lower": lower,
            "upper": upper,
            "mean": self.posterior.mean(),
            "std": self.posterior_std()
        }


class GaussianGaussian:
    def __init__(self, prior_mu: float, prior_sigma: float, obs_sigma: float):
        if prior_sigma <= 0.0 or obs_sigma <= 0.0:
            raise ValueError("Standard deviations must be positive.")
        self.prior_mu = prior_mu
        self.prior_var = prior_sigma ** 2
        self.obs_var = obs_sigma ** 2
        self.prior = NormalDistribution(prior_mu, prior_sigma)
        self.posterior = self.prior          

    def update(self, data: list):
        n = len(data)
        if n == 0:
            raise ValueError("data list must not be empty.")
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

    def map_estimate(self):
        """Normal MAP is the mean (mode = mean)."""
        return self.posterior.mu

    def credible_interval(self, prob: float = 0.95) -> dict:
        norm = NormalDistribution(0.0, 1.0)
        z = norm.ppf((1.0 + prob) / 2.0)
        mu = self.posterior.mu
        std = self.posterior.sigma
        lower = mu - z * std
        upper = mu + z * std
        return {
            "prob": prob,
            "lower": lower,
            "upper": upper,
            "mean": mu,
            "std": std,
        }

