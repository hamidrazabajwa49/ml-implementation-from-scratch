import math
import random
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Probability.distributions import Distribution
from special_functions import betainc, gammainc

class NormalDistribution:
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x)."""
        z = (x - self.mu) / self.sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def pdf(self, x: float) -> float:
        """Probability density function."""
        z = (x - self.mu) / self.sigma
        return math.exp(-0.5 * z ** 2) / (self.sigma * math.sqrt(2.0 * math.pi))

    def mean(self):
        return self.mu

    def variance(self):
        return self.sigma ** 2
    
    def sf(self, x: float) -> float:
        """Survival function P(X > x)."""
        return 1.0 - self.cdf(x)

    def ppf(self, p: float) -> float:
        """
        Percent-point function (quantile function / inverse CDF).

        Returns the value x such that P(X <= x) = p.

        Uses the Beasley-Springer-Moro rational approximation.
        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError("p must be strictly between 0 and 1")

        a = [
            -3.969683028665376e+01,  2.209460984245205e+02,
            -2.759285104469687e+02,  1.383577518672690e+02,
            -3.066479806614716e+01,  2.506628277459239e+00,
        ]
        b = [
            -5.447609879822406e+01,  1.615858368580409e+02,
            -1.556989798598866e+02,  6.680131188771972e+01,
            -1.328068155288572e+01,
        ]
        c = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00,  2.938163982698783e+00,
        ]
        d = [
            7.784695709041462e-03,  3.224671290700398e-01,
            2.445134137142996e+00,  3.754408661907416e+00,
        ]

        p_low = 0.02425
        p_high = 1.0 - p_low

        if p < p_low:
            q = math.sqrt(-2.0 * math.log(p))
            z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            z = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        else:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

        return self.mu + self.sigma * z


class TDistribution:
    def __init__(self, df: float):
        if df <= 0.0:
            raise ValueError("df must be positive")
        self.df = df

    def pdf(self, t: float) -> float:
        """Probability density function."""
        coeff = math.exp(
            math.lgamma((self.df + 1.0) / 2.0) - math.lgamma(self.df / 2.0)
        )
        coeff /= math.sqrt(self.df * math.pi)
        return coeff * (1.0 + t ** 2 / self.df) ** (-(self.df + 1.0) / 2.0)

    def cdf(self, t: float) -> float:
        """Cumulative distribution function P(X <= t)."""
        x = self.df / (self.df + t ** 2)
        p = 0.5 * betainc(self.df / 2.0, 0.5, x)
        if t > 0:
            return 1.0 - p
        return p

    def sf(self, t: float) -> float:
        """Survival function P(X > t)."""
        return 1.0 - self.cdf(t)

    def ppf(self, p: float) -> float:
        """
        Percent-point function (quantile function / inverse CDF).

        Returns the value t such that P(X <= t) = p.
        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError("p must be strictly between 0 and 1")
        lo, hi = -1.0, 1.0
        while self.cdf(lo) >= p:
            lo *= 2.0
        while self.cdf(hi) <= p:
            hi *= 2.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def p_value(self, t: float, alternative: str) -> float:
        """
        Compute the p-value for a given t-statistic.
        """
        if alternative == "two-sided":
            return 2.0 * min(self.cdf(t), self.sf(t))
        if alternative == "greater":
            return self.sf(t)
        if alternative == "less":
            return self.cdf(t)
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


class Chi2Distribution:
    def __init__(self, df: float):
        if df <= 0.0:
            raise ValueError("df must be positive")
        self.df = df

    def pdf(self, x: float) -> float:
        """Probability density function."""
        if x <= 0.0:
            return 0.0
        k = self.df / 2.0
        return math.exp(
            (k - 1.0) * math.log(x) - x / 2.0 - k * math.log(2.0) - math.lgamma(k)
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x)."""
        if x <= 0.0:
            return 0.0
        return gammainc(self.df / 2.0, x / 2.0)

    def sf(self, x: float) -> float:
        """Survival function P(X > x)."""
        return 1.0 - self.cdf(x)



class FDistribution:
    
    def __init__(self, df1: float, df2: float):
        if df1 <= 0.0:
            raise ValueError("df1 must be positive")
        if df2 <= 0.0:
            raise ValueError("df2 must be positive")
        self.df1 = df1
        self.df2 = df2

    def pdf(self, f: float) -> float:
        """Probability density function."""
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
        """Cumulative distribution function P(X <= f)."""
        if f <= 0.0:
            return 0.0
        x = self.df1 * f / (self.df1 * f + self.df2)
        return betainc(self.df1 / 2.0, self.df2 / 2.0, x)

    def sf(self, f: float) -> float:
        """Survival function P(X > f)."""
        return 1.0 - self.cdf(f)

class BetaDistribution:
    def __init__(self,alpha:float,beta:float):
        if (alpha<0 or beta<0):
            raise ValueError ("parameters must be greater than zero.")
        self.alpha=alpha
        self.beta=beta

    def mean(self):
        return self.alpha/(self.alpha + self.beta)

    def variance(self):
        denom=(self.alpha+self.beta)**2 * (self.alpha + self.beta + 1)
        return (self.alpha * self.beta)/denom

    def beta(alpha,beta):
        num=math.gamma(alpha)*math.gamma(beta)
        denom=math.gamma((alpha+beta))
        return num/denom

    def pdf(self,x):
        denom=BetaDistribution.beta(self.alpha,self.beta)
        num=(x**(self.alpha-1))*((1-x)**(self.beta-1))
        return num/denom

    def sample(self):
        return [self.mean()]


class GammaDistribution:

    def __init__(self,alpha:float,beta:float):
        if (alpha<0 or beta<0):
            raise ValueError("parameters must be greater than zero.")
        self.alpha=alpha
        self.beta=beta

    def mean(self):
        return self.alpha/self.beta

    def variance(self):
        return self.alpha/(self.beta**2)

    def pdf(self,x):
        if (x<0):
            return 0.0
        num=(self.beta**self.alpha)*(x**(self.alpha-1))*(math.exp(-self.beta*x))
        return num/math.gamma(self.alpha)

    def sample(self,num_samples=1):
        return [self.mean()]*num_samples
