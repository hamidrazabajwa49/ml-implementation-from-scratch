import math
import random
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from Probability.distributions import Distribution

class NormalDistribution(Distribution):
    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("Standard deviation sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma ** 2

    def pdf(self, x: float) -> float:
        coeff = 1.0 / (self.sigma * math.sqrt(2.0 * math.pi))
        exponent = -((x - self.mu) ** 2) / (2.0 * self.sigma ** 2)
        return coeff * math.exp(exponent)

    def cdf(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def sample(self, num_samples: int = 1) -> list:
        samples = []
        needed = (num_samples + 1) // 2  
        for _ in range(needed):
            u1 = random.random()
            u2 = random.random()
            r = math.sqrt(-2.0 * math.log(u1))
            theta = 2.0 * math.pi * u2
            z1 = r * math.cos(theta)
            z2 = r * math.sin(theta)
            samples.append(z1 * self.sigma + self.mu)
            samples.append(z2 * self.sigma + self.mu)
        return samples[:num_samples]
