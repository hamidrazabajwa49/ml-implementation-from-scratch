import math
import random
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Probability.distributions import Distribution


class BinomialDistribution(Distribution):
    def __init__(self, n: int, p: float):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be between 0 and 1.")
        self.n = n
        self.p = p

    def mean(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1.0 - self.p)

    def pmf(self, k: int) -> float:
        if not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if not (0 <= k <= self.n):
            return 0.0
        return math.comb(self.n, k) * (self.p ** k) * ((1.0 - self.p) ** (self.n - k))

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        if x >= self.n:
            return 1.0
        return sum(self.pmf(k) for k in range(int(x) + 1))

    def sample(self, num_samples: int = 1) -> list:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        return [sum(1 for _ in range(self.n) if random.random() < self.p)
                for _ in range(num_samples)]


class BernoulliDistribution(Distribution):
    def __init__(self, p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be between 0 and 1.")
        self.p = p

    def mean(self) -> float:
        return self.p

    def variance(self) -> float:
        return self.p * (1.0 - self.p)

    def pmf(self, x: int) -> float:
        if x == 0:
            return 1.0 - self.p
        if x == 1:
            return self.p
        return 0.0

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        if x < 1:
            return 1.0 - self.p
        return 1.0

    def sample(self, num_samples: int = 1) -> list:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        return [1 if random.random() < self.p else 0 for _ in range(num_samples)]


class PoissonDistribution(Distribution):
    def __init__(self, lam: float):
        if lam < 0.0:
            raise ValueError("Lambda must be non-negative.")
        self.lam = lam

    def mean(self) -> float:
        return self.lam

    def variance(self) -> float:
        return self.lam

    def pmf(self, k: int) -> float:
        if not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if k < 0:
            return 0.0
        if self.lam == 0.0:
            return 1.0 if k == 0 else 0.0
        return (self.lam ** k) * math.exp(-self.lam) / math.factorial(k)

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return sum(self.pmf(k) for k in range(int(x) + 1))

    def sample(self, num_samples: int = 1) -> list:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        if self.lam == 0.0:
            return [0] * num_samples
        samples = []
        L = math.exp(-self.lam)
        for _ in range(num_samples):
            p = 1.0
            k = 0
            while p > L:
                k += 1
                p *= random.random()
            samples.append(k - 1)
        return samples
