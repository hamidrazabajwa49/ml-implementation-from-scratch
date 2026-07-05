"""
continuous.py

Continuous probability distributions: Normal, Student-t, Chi-squared, F,
Beta, and Gamma. Each subclasses :class:`distributions.Distribution` and
exposes ``pdf``/``cdf``/``sf``/``ppf``/``mean``/``variance``/``sample``
where mathematically defined.

CDFs for Beta/Gamma are computed via the *exact* regularized incomplete
beta/gamma functions in :mod:`special_functions` (continued fractions),
not numerical integration -- this is both dramatically faster and more
accurate than trapezoidal quadrature.

Example
>>> n = NormalDistribution(mu=0, sigma=1)
>>> round(n.cdf(1.96), 4)
0.975
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
from Probability.distributions import Distribution  
from Probability.special_functions import betainc, gammainc  


def _check_positive(value: float, name: str) -> None:
    """Validate that ``value`` is a positive, finite, non-NaN real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    if math.isinf(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


class NormalDistribution(Distribution):
    """A Normal(mu, sigma) distribution.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation; must be positive.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if isinstance(mu, bool) or not isinstance(mu, (int, float)) or math.isnan(mu):
            raise ValueError("mu must be a finite real number")
        _check_positive(sigma, "sigma")
        self.mu = float(mu)
        self.sigma = float(sigma)

    def __repr__(self) -> str:
        return f"NormalDistribution(mu={self.mu}, sigma={self.sigma})"

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma ** 2

    def pdf(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return math.exp(-0.5 * z ** 2) / (self.sigma * math.sqrt(2.0 * math.pi))

    def cdf(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def sf(self, x: float) -> float:
        """P(X > x), computed via erfc for accuracy far in the tails."""
        z = (x - self.mu) / self.sigma
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    def ppf(self, p: float) -> float:
        """Inverse CDF via the Acklam algorithm, refined with one Halley step.

        Raises
        ------
        ValueError
            If ``p`` is not strictly between 0 and 1.
        """
        if not (0.0 < p < 1.0):
            raise ValueError("p must be strictly between 0 and 1")

        a = [
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00,
        ]
        b = [
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01,
        ]
        c = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00,
        ]
        d = [
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00,
        ]

        p_low = 0.02425
        p_high = 1.0 - p_low

        if p < p_low:
            q = math.sqrt(-2.0 * math.log(p))
            z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        else:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

        # One Halley refinement step pushes Acklam's ~1.15e-9 relative
        # Error down to ~machine precision, at negligible extra cost
        e = 0.5 * math.erfc(-z / math.sqrt(2.0)) - p
        u = e * math.sqrt(2.0 * math.pi) * math.exp(z * z / 2.0)
        z = z - u / (1.0 + z * u / 2.0)

        return self.mu + self.sigma * z

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[float]:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        rng = random.Random(seed) if seed is not None else random
        return [rng.gauss(self.mu, self.sigma) for _ in range(num_samples)]


class TDistribution(Distribution):
    """Student's t-distribution with ``df`` degrees of freedom.

    Parameters
    ----------
    df : float
        Degrees of freedom; must be positive.
    """

    def __init__(self, df: float):
        _check_positive(df, "df")
        self.df = float(df)

    def __repr__(self) -> str:
        return f"TDistribution(df={self.df})"

    def mean(self) -> float:
        """Mean, defined only for df > 1.

        Raises
        ------
        ValueError
            If ``df <= 1`` (mean is undefined).
        """
        if self.df <= 1.0:
            raise ValueError(f"Mean is undefined for df <= 1 (got df={self.df})")
        return 0.0

    def variance(self) -> float:
        """Variance: ``df / (df - 2)`` for df > 2, infinite for 1 < df <= 2.

        Raises
        ------
        ValueError
            If ``df <= 1`` (variance is undefined).
        """
        if self.df <= 1.0:
            raise ValueError(f"Variance is undefined for df <= 1 (got df={self.df})")
        if self.df <= 2.0:
            return math.inf
        return self.df / (self.df - 2.0)

    def pdf(self, t: float) -> float:
        coeff = math.exp(
            math.lgamma((self.df + 1.0) / 2.0) - math.lgamma(self.df / 2.0)
        ) / math.sqrt(self.df * math.pi)
        return coeff * (1.0 + t ** 2 / self.df) ** (-(self.df + 1.0) / 2.0)

    def cdf(self, t: float) -> float:
        x = self.df / (self.df + t ** 2)
        p = 0.5 * betainc(self.df / 2.0, 0.5, x)
        return p if t <= 0 else 1.0 - p

    def sf(self, t: float) -> float:
        return 1.0 - self.cdf(t)

    def ppf(self, p: float, tol: float = 1e-12, max_expand: int = 200) -> float:
        """Inverse CDF via bracketing + bisection (t has no closed-form ppf).

        Raises
        ------
        ValueError
            If ``p`` is not strictly between 0 and 1.
        RuntimeError
            If a valid bracket cannot be found within ``max_expand`` doublings.
        """
        if not (0.0 < p < 1.0):
            raise ValueError("p must be strictly between 0 and 1")
        lo, hi = -1.0, 1.0
        expansions = 0
        while self.cdf(lo) >= p:
            lo *= 2.0
            expansions += 1
            if expansions > max_expand:
                raise RuntimeError("Could not bracket the root; p may be too extreme.")
        expansions = 0
        while self.cdf(hi) <= p:
            hi *= 2.0
            expansions += 1
            if expansions > max_expand:
                raise RuntimeError("Could not bracket the root; p may be too extreme.")
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if hi - lo < tol:
                break
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def p_value(self, t: float, alternative: str) -> float:
        """One- or two-sided p-value for observed statistic ``t``.

        Parameters
        ----------
        alternative : {'two-sided', 'greater', 'less'}
        """
        if alternative == "two-sided":
            return 2.0 * min(self.cdf(t), self.sf(t))
        if alternative == "greater":
            return self.sf(t)
        if alternative == "less":
            return self.cdf(t)
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[float]:
        """Sample via ``Z / sqrt(V / df)`` where ``Z ~ N(0,1)``, ``V ~ Chi2(df)``."""
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        rng = random.Random(seed) if seed is not None else random
        chi2 = Chi2Distribution(self.df)
        z_samples = [rng.gauss(0.0, 1.0) for _ in range(num_samples)]
        v_samples = chi2.sample(num_samples, seed=rng.randint(0, 2**31 - 1))
        return [z / math.sqrt(v / self.df) for z, v in zip(z_samples, v_samples)]


class Chi2Distribution(Distribution):
    """Chi-squared distribution with ``df`` degrees of freedom.

    Parameters
    ----------
    df : float
        Degrees of freedom; must be positive.
    """

    def __init__(self, df: float):
        _check_positive(df, "df")
        self.df = float(df)

    def __repr__(self) -> str:
        return f"Chi2Distribution(df={self.df})"

    def mean(self) -> float:
        return self.df

    def variance(self) -> float:
        return 2.0 * self.df

    def pdf(self, x: float) -> float:
        if x < 0.0:
            return 0.0
        k = self.df / 2.0
        if x == 0.0:
            if k > 1.0:
                return 0.0
            if k == 1.0:
                return 0.5
            return math.inf
        return math.exp(
            (k - 1.0) * math.log(x) - x / 2.0 - k * math.log(2.0) - math.lgamma(k)
        )

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        return gammainc(self.df / 2.0, x / 2.0)

    def sf(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def ppf(self, p: float, tol: float = 1e-12) -> float:
        """Inverse CDF via bracketing + bisection."""
        if not (0.0 < p < 1.0):
            raise ValueError("p must be strictly between 0 and 1")
        upper = self.mean() + 4.0 * self.std() + 1.0
        expansions = 0
        while self.cdf(upper) < p:
            upper *= 2.0
            expansions += 1
            if expansions > 200:
                raise RuntimeError("Could not bracket the root; p may be too extreme.")
        lo, hi = 0.0, upper
        for _ in range(100):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2.0
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[float]:
        """Sample via the Chi2(df) = Gamma(shape=df/2, rate=1/2) relationship."""
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        gamma = GammaDistribution(alpha=self.df / 2.0, beta=0.5)
        return gamma.sample(num_samples, seed=seed)


class FDistribution(Distribution):
    """F-distribution with ``df1`` (numerator) and ``df2`` (denominator) degrees of freedom.

    Parameters
    ----------
    df1, df2 : float
        Degrees of freedom; must be positive.
    """

    def __init__(self, df1: float, df2: float):
        _check_positive(df1, "df1")
        _check_positive(df2, "df2")
        self.df1 = float(df1)
        self.df2 = float(df2)

    def __repr__(self) -> str:
        return f"FDistribution(df1={self.df1}, df2={self.df2})"

    def mean(self) -> float:
        """Mean, defined only for df2 > 2."""
        if self.df2 <= 2.0:
            raise ValueError(f"Mean is undefined for df2 <= 2 (got df2={self.df2})")
        return self.df2 / (self.df2 - 2.0)

    def variance(self) -> float:
        """Variance, defined only for df2 > 4."""
        if self.df2 <= 4.0:
            raise ValueError(f"Variance is undefined for df2 <= 4 (got df2={self.df2})")
        d1, d2 = self.df1, self.df2
        return (2.0 * d2 ** 2 * (d1 + d2 - 2.0)) / (d1 * (d2 - 2.0) ** 2 * (d2 - 4.0))

    def pdf(self, f: float) -> float:
        if f <= 0.0:
            return 0.0
        d1, d2 = self.df1, self.df2
        log_num = (
            (d1 / 2.0) * math.log(d1)
            + (d2 / 2.0) * math.log(d2)
            + (d1 / 2.0 - 1.0) * math.log(f)
        )
        log_den = (
            math.lgamma(d1 / 2.0)
            + math.lgamma(d2 / 2.0)
            - math.lgamma((d1 + d2) / 2.0)
            + ((d1 + d2) / 2.0) * math.log(d2 + d1 * f)
        )
        return math.exp(log_num - log_den)

    def cdf(self, f: float) -> float:
        if f <= 0.0:
            return 0.0
        x = self.df1 * f / (self.df1 * f + self.df2)
        return betainc(self.df1 / 2.0, self.df2 / 2.0, x)

    def sf(self, f: float) -> float:
        return 1.0 - self.cdf(f)

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[float]:
        """Sample via ``(V1/df1) / (V2/df2)`` where ``V1 ~ Chi2(df1)``, ``V2 ~ Chi2(df2)``."""
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        rng = random.Random(seed) if seed is not None else random
        v1 = Chi2Distribution(self.df1).sample(num_samples, seed=rng.randint(0, 2**31 - 1))
        v2 = Chi2Distribution(self.df2).sample(num_samples, seed=rng.randint(0, 2**31 - 1))
        return [(a / self.df1) / (b / self.df2) for a, b in zip(v1, v2)]


class BetaDistribution(Distribution):
    """Beta(alpha, beta) distribution on ``[0, 1]``.

    Parameters
    ----------
    alpha, beta : float
        Shape parameters; must be positive.
    """

    def __init__(self, alpha: float, beta: float):
        _check_positive(alpha, "alpha")
        _check_positive(beta, "beta")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __repr__(self) -> str:
        return f"BetaDistribution(alpha={self.alpha}, beta={self.beta})"

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s ** 2 * (s + 1.0))

    @staticmethod
    def beta_func(alpha: float, beta: float) -> float:
        """The (non-regularized) Beta function ``B(alpha, beta)``."""
        return math.exp(math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))

    def pdf(self, x: float) -> float:
        if x < 0.0 or x > 1.0:
            return 0.0
        if x == 0.0:
            return 0.0 if self.alpha >= 1.0 else math.inf
        if x == 1.0:
            return 0.0 if self.beta >= 1.0 else math.inf
        log_pdf = (
            (self.alpha - 1.0) * math.log(x)
            + (self.beta - 1.0) * math.log(1.0 - x)
            - (math.lgamma(self.alpha) + math.lgamma(self.beta) - math.lgamma(self.alpha + self.beta))
        )
        return math.exp(log_pdf)

    def cdf(self, x: float) -> float:
        """Exact CDF via the regularized incomplete beta function (see :func:`betainc`)."""
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return betainc(self.alpha, self.beta, x)

    def sf(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def ppf(self, p: float, tol: float = 1e-12) -> float:
        """Inverse CDF via bracketing + bisection."""
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(100):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2.0
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[float]:
        """Sample via Johnk's rejection method (valid for all alpha, beta > 0).

        Note
        ----
        Acceptance rate degrades for large alpha/beta; this is a known
        limitation of a from-scratch, NumPy-free implementation.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        rng = random.Random(seed) if seed is not None else random
        results = []
        for _ in range(num_samples):
            while True:
                u = rng.random() ** (1.0 / self.alpha)
                v = rng.random() ** (1.0 / self.beta)
                if u + v <= 1.0:
                    results.append(u / (u + v))
                    break
        return results


class GammaDistribution(Distribution):
    """Gamma(alpha, beta) distribution (shape-rate parameterization).

    ``pdf(x) = beta^alpha * x^(alpha-1) * exp(-beta*x) / Gamma(alpha)``

    Parameters
    ----------
    alpha : float
        Shape parameter; must be positive.
    beta : float
        Rate parameter; must be positive.
    """

    def __init__(self, alpha: float, beta: float):
        _check_positive(alpha, "alpha")
        _check_positive(beta, "beta")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __repr__(self) -> str:
        return f"GammaDistribution(alpha={self.alpha}, beta={self.beta})"

    def mean(self) -> float:
        return self.alpha / self.beta

    def variance(self) -> float:
        return self.alpha / (self.beta ** 2)

    def pdf(self, x: float) -> float:
        if x < 0.0:
            return 0.0
        if x == 0.0:
            return 0.0 if self.alpha >= 1.0 else math.inf
        log_pdf = (
            self.alpha * math.log(self.beta)
            + (self.alpha - 1.0) * math.log(x)
            - self.beta * x
            - math.lgamma(self.alpha)
        )
        return math.exp(log_pdf)

    def cdf(self, x: float) -> float:
        """Exact CDF via the regularized lower incomplete gamma function (see :func:`gammainc`)."""
        if x <= 0.0:
            return 0.0
        return gammainc(self.alpha, self.beta * x)

    def sf(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def ppf(self, p: float, tol: float = 1e-12) -> float:
        """Inverse CDF via bracketing + bisection."""
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return float("inf")
        upper = self.mean() + 4.0 * self.std() + 1.0
        expansions = 0
        while self.cdf(upper) < p:
            upper *= 2.0
            expansions += 1
            if expansions > 200:
                raise RuntimeError("Could not bracket the root; p may be too extreme.")
        lo, hi = 0.0, upper
        for _ in range(100):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2.0
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[float]:
        """Sample via the Marsaglia-Tsang method (boosted for alpha < 1)."""
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        rng = random.Random(seed) if seed is not None else random

        results = []
        alpha = self.alpha
        boost = alpha < 1.0
        if boost:
            alpha = alpha + 1.0
        d = alpha - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        for _ in range(num_samples):
            while True:
                x = rng.gauss(0.0, 1.0)
                v = (1.0 + c * x) ** 3
                if v <= 0.0:
                    continue
                u = rng.random()
                x4 = x ** 4
                if u < 1.0 - 0.0331 * x4:
                    s = d * v
                    break
                if math.log(u) < 0.5 * x ** 2 + d * (1.0 - v + math.log(v)):
                    s = d * v
                    break
            if boost:
                s *= rng.random() ** (1.0 / self.alpha)
            results.append(s / self.beta)
        return results
