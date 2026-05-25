import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Probability.continuous import NormalDistribution,BetaDistribution,GammaDistribution

class BetaBinomial:
    
    def __init__(self, alpha_prior, beta_prior):
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.prior = BetaDistribution(alpha_prior, beta_prior)

    def update(self, n_trials: int, k_success: int):
        self.alpha += k_success
        self.beta += (n_trials - k_success)
        self.posterior = BetaDistribution(self.alpha, self.beta)

    def posterior_mean(self):
        return self.posterior.mean()

    def credible_interval(self, prob=0.95):
        # Get z from standard normal using your ppf
        norm = NormalDistribution(0.0, 1.0)
        z = norm.ppf((1.0 + prob) / 2.0)

        mu = self.posterior.mean()
        std = math.sqrt(self.posterior.variance())
        lower = mu - z * std
        upper = mu + z * std
        
        return {
            "prob": prob,
            "lower": lower,
            "upper": upper,
            "mean": mu,
            "std": std
        }

class GammaPoisson:

    def __init__(self,alpha_prior,beta_prior):
        self.alpha=alpha_prior
        self.beta=beta_prior
        self.prior=GammaDistribution(alpha_prior,beta_prior)

    def update(self,k_events,t_exposure=1.0):
        self.alpha+=k_events
        self.beta+=t_exposure
        self.posterior=GammaDistribution(self.alpha,self.beta)

    def posterior_mean(self):
        return self.posterior.mean()

    def credible_interval(self,prob=0.95):
        norm = NormalDistribution(0.0, 1.0)
        z = norm.ppf((1.0 + prob) / 2.0)
        mu = self.posterior.mean()
        std = math.sqrt(self.posterior.variance())
        lower = mu - z * std
        upper = mu + z * std
        
        return {
            "prob": prob,
            "lower": lower,
            "upper": upper,
            "mean": mu,
            "std": std
        }


class GaussianGaussian:

    def __init__(self, prior_mu: float, prior_sigma: float, obs_sigma: float):
        if (prior_sigma <= 0.0 or obs_sigma <= 0.0):
            raise ValueError("Standard deviations must be positive.")
        self.prior_mu = prior_mu
        self.prior_var = prior_sigma ** 2
        self.obs_var = obs_sigma ** 2
        self.prior = NormalDistribution(prior_mu, prior_sigma)

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

        self.posterior = NormalDistribution(post_mu, math.sqrt(post_var))

    def posterior_mean(self) -> float:
        return self.posterior.mean()

    def credible_interval(self, prob: float = 0.95) -> dict:
        norm = NormalDistribution(0.0, 1.0)
        z = norm.ppf((1.0 + prob) / 2.0)
        mu = self.posterior.mean()
        std = math.sqrt(self.posterior.variance())
        lower = mu - z * std
        upper = mu + z * std
        
        return {
            "prob": prob,
            "lower": lower,
            "upper": upper,
            "mean": mu,
            "std": std,
        }
