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
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")
        self.n = n
        self.p = p
        self._params = {'n': n, 'p': p}

    def mean(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)

    def pmf(self, k: int) -> float:
        if not (0 <= k <= self.n):
            return 0.0
        comb = math.comb(self.n, k)
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        if x >= self.n:
            return 1.0
        # Sum pmf from k=0 to floor(x)
        total = 0.0
        for k in range(int(x) + 1):
            total += self.pmf(k)
        return total

    def sample(self, num_samples: int = 1) -> list:
        samples = []
        for _ in range(num_samples):
            successes = sum(1 for _ in range(self.n) if random.random() < self.p)
            samples.append(successes)
        return samples


class BernoulliDistribution(Distribution):
    def __init__(self, p: float):
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")
        self.p = p
        self._params = {'p': p}

    def mean(self) -> float:
        return self.p

    def variance(self) -> float:
        return self.p * (1 - self.p)

    def pmf(self, x: int) -> float:
        if x == 0:
            return 1 - self.p
        elif x == 1:
            return self.p
        else:
            return 0.0

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        elif x < 1:
            return 1 - self.p
        else:
            return 1.0

    def sample(self, num_samples: int = 1) -> list:
        samples = []
        for _ in range(num_samples):
            samples.append(1 if random.random() < self.p else 0)
        return samples

class PoissonDistribution(Distribution):
    def __init__(self, lam: float):
        if lam < 0.0:
            raise ValueError("Lambda must be non-negative.")
        self.lam = lam
        self._params = {'lam': lam}

    def mean(self) -> float:
        return self.lam

    def variance(self) -> float:
        return self.lam

    def pmf(self, k: int) -> float:
        if k < 0:
            return 0.0
        return (self.lam ** k) * math.exp(-self.lam) / math.factorial(k)

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        total = 0.0
        for k in range(int(x) + 1):
            total += self.pmf(k)
        return total

    def sample(self, num_samples: int = 1) -> list:
        samples = []
        for _ in range(num_samples):
            L = math.exp(-self.lam)
            p = 1.0
            k = 0
            while p > L:
                k += 1
                p *= random.random()
            samples.append(k - 1)
        return samples
