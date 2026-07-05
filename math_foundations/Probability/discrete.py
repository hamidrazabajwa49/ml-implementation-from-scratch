"""
discrete.py
===========

Discrete probability distributions: Bernoulli, Binomial, and Poisson.
Each subclasses :class:`distributions.Distribution` and exposes the
standard ``pmf``/``cdf``/``sf``/``ppf``/``mean``/``variance``/``sample``
interface.

Example
-------
>>> b = BinomialDistribution(n=10, p=0.3)
>>> round(b.pmf(3), 4)
0.2668
>>> round(b.cdf(3), 4)
0.6496
"""

from __future__ import annotations

import os
import sys
import math
import random
from typing import List, Optional


_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from Probability.distributions import Distribution  # type: ignore


def _check_probability(p: float, name: str = "p") -> None:
    """Validate that ``p`` is a real number in ``[0, 1]``."""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(p).__name__}")
    if math.isnan(p):
        raise ValueError(f"{name} must not be NaN")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {p}")


def _check_nonneg_int(n: int, name: str = "n") -> None:
    """Validate that ``n`` is a non-negative ``int`` (bools excluded)."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"{name} must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"{name} must be non-negative, got {n}")


class BernoulliDistribution(Distribution):
    """A Bernoulli(p) distribution over ``{0, 1}``.

    Parameters
    ----------
    p : float
        Probability of success (outcome 1); must be in ``[0, 1]``.
    """

    def __init__(self, p: float):
        _check_probability(p)
        self.p = float(p)

    def __repr__(self) -> str:
        return f"BernoulliDistribution(p={self.p})"

    def mean(self) -> float:
        return self.p

    def variance(self) -> float:
        return self.p * (1.0 - self.p)

    def pmf(self, x: int) -> float:
        """P(X = x)."""
        if x == 0:
            return 1.0 - self.p
        if x == 1:
            return self.p
        return 0.0

    def cdf(self, x: float) -> float:
        """P(X <= x)."""
        if x < 0:
            return 0.0
        if x < 1:
            return 1.0 - self.p
        return 1.0

    def sf(self, x: float) -> float:
        """P(X > x)."""
        return 1.0 - self.cdf(x)

    def ppf(self, q: float) -> int:
        """Smallest integer ``x`` such that ``cdf(x) >= q``."""
        if not (0.0 < q <= 1.0):
            raise ValueError("q must be in (0, 1]")
        return 0 if q <= 1.0 - self.p else 1

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[int]:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        rng = random.Random(seed) if seed is not None else random
        return [1 if rng.random() < self.p else 0 for _ in range(num_samples)]


class BinomialDistribution(Distribution):
    """A Binomial(n, p) distribution: the number of successes in ``n`` i.i.d. Bernoulli(p) trials.

    Parameters
    ----------
    n : int
        Number of trials; must be a non-negative integer.
    p : float
        Success probability per trial; must be in ``[0, 1]``.
    """

    def __init__(self, n: int, p: float):
        _check_nonneg_int(n, "n")
        _check_probability(p)
        self.n = n
        self.p = float(p)

    def __repr__(self) -> str:
        return f"BinomialDistribution(n={self.n}, p={self.p})"

    def mean(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1.0 - self.p)

    def pmf(self, k: int) -> float:
        """P(X = k)."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an int.")
        if not (0 <= k <= self.n):
            return 0.0
        return math.comb(self.n, k) * (self.p ** k) * ((1.0 - self.p) ** (self.n - k))

    def cdf(self, x: float) -> float:
        """P(X <= x). O(x) via direct summation; fine for typical n."""
        if x < 0:
            return 0.0
        if x >= self.n:
            return 1.0
        return sum(self.pmf(k) for k in range(int(x) + 1))

    def sf(self, x: float) -> float:
        """P(X > x)."""
        return 1.0 - self.cdf(x)

    def ppf(self, q: float) -> int:
        """Smallest integer ``k`` such that ``cdf(k) >= q``."""
        if not (0.0 < q <= 1.0):
            raise ValueError("q must be in (0, 1]")
        cumulative = 0.0
        for k in range(self.n + 1):
            cumulative += self.pmf(k)
            if cumulative >= q:
                return k
        return self.n

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[int]:
        """Draw samples by simulating ``n`` Bernoulli trials per sample: O(num_samples * n).

        For very large ``n`` this is the dominant cost; a from-scratch
        implementation without NumPy has no cheaper exact alternative
        that stays this simple and numerically robust.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        rng = random.Random(seed) if seed is not None else random
        return [
            sum(1 for _ in range(self.n) if rng.random() < self.p)
            for _ in range(num_samples)
        ]


class PoissonDistribution(Distribution):
    """A Poisson(lambda) distribution over the non-negative integers.

    Parameters
    ----------
    lam : float
        Rate parameter; must be non-negative.
    """

    #: Above this rate, Knuth's direct-simulation algorithm underflows
    #: (exp(-lam) rounds to 0.0) and would loop effectively forever; we
    #: decompose into independent chunks below this size instead.
    _KNUTH_SAFE_LAMBDA = 20.0

    def __init__(self, lam: float):
        if isinstance(lam, bool) or not isinstance(lam, (int, float)):
            raise TypeError(f"lam must be a real number, got {type(lam).__name__}")
        if math.isnan(lam):
            raise ValueError("lam must not be NaN")
        if math.isinf(lam):
            raise ValueError("lam must be finite")
        if lam < 0.0:
            raise ValueError("lam must be non-negative.")
        self.lam = float(lam)

    def __repr__(self) -> str:
        return f"PoissonDistribution(lam={self.lam})"

    def mean(self) -> float:
        return self.lam

    def variance(self) -> float:
        return self.lam

    def pmf(self, k: int) -> float:
        """P(X = k)."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if k < 0:
            return 0.0
        if self.lam == 0.0:
            return 1.0 if k == 0 else 0.0
        # log-space to avoid overflow in lam**k / k! for large k or lam.
        log_pmf = k * math.log(self.lam) - self.lam - math.lgamma(k + 1)
        return math.exp(log_pmf)

    def cdf(self, x: float) -> float:
        """P(X <= x). O(x) via direct summation; fine for typical lambda."""
        if x < 0:
            return 0.0
        return sum(self.pmf(k) for k in range(int(x) + 1))

    def sf(self, x: float) -> float:
        """P(X > x)."""
        return 1.0 - self.cdf(x)

    def ppf(self, q: float, max_k: int = 1_000_000) -> int:
        """Smallest integer ``k`` such that ``cdf(k) >= q``.

        Parameters
        ----------
        max_k : int, optional
            Safety cap on the search to guarantee termination for
            pathological inputs.
        """
        if not (0.0 < q <= 1.0):
            raise ValueError("q must be in (0, 1]")
        cumulative = 0.0
        k = 0
        while k <= max_k:
            cumulative += self.pmf(k)
            if cumulative >= q:
                return k
            k += 1
        raise RuntimeError(f"ppf did not converge within max_k={max_k} for lam={self.lam}")

    def _sample_knuth(self, rng: random.Random) -> int:
        """Knuth's direct-simulation algorithm; only safe for lam <= _KNUTH_SAFE_LAMBDA."""
        L = math.exp(-self.lam)
        p = 1.0
        k = 0
        while p > L:
            k += 1
            p *= rng.random()
        return k - 1

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[int]:
        """Draw samples via Knuth's algorithm.

        For ``lam`` above ~20, ``exp(-lam)`` underflows to exactly 0.0 in
        floating point, which would make Knuth's algorithm loop far too
        long (or effectively forever). We exploit the fact that a
        Poisson(lam) variate is the sum of independent Poisson(lam/m)
        variates (infinite divisibility) and decompose large rates into
        safe-sized chunks.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        rng = random.Random(seed) if seed is not None else random
        if self.lam == 0.0:
            return [0] * num_samples

        if self.lam <= self._KNUTH_SAFE_LAMBDA:
            return [self._sample_knuth(rng) for _ in range(num_samples)]

        n_chunks = math.ceil(self.lam / self._KNUTH_SAFE_LAMBDA)
        chunk_lam = self.lam / n_chunks
        chunk = PoissonDistribution(chunk_lam)
        results = []
        for _ in range(num_samples):
            total = 0
            for _ in range(n_chunks):
                total += chunk._sample_knuth(rng)
            results.append(total)
        return results
