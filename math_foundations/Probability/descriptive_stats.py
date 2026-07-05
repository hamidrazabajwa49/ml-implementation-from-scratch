"""
descriptive_stats.py
=====================

Descriptive statistics for a single dataset (mean, median, mode,
percentiles, spread, shape) plus paired/multivariate summaries
(covariance, correlation, covariance/correlation matrices). Integrates
with :mod:`matrix` (Module 2) for the matrix-valued outputs.

Example
-------
>>> s = DescriptiveStats([1, 2, 2, 3, 4])
>>> s.mean()
2.4
>>> s.mode()
[2]
"""

from __future__ import annotations

import os
import sys
import math
from collections import Counter
from typing import List, Sequence, Union



_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from Matrix.matrix import Matrix  # type: ignore

Number = Union[int, float]


def _validate_numeric_data(data: Sequence[Number], name: str = "data") -> None:
    """Validate that ``data`` is a non-empty sequence of finite, non-NaN numbers.

    Raises
    ------
    ValueError
        If ``data`` is empty, or contains NaN (sorting/ordering-based
        statistics like median/percentile are undefined with NaN present,
        since NaN comparisons are always False in IEEE-754/Python).
    TypeError
        If any element is not an ``int``/``float`` (``bool`` excluded).
    """
    if not data:
        raise ValueError(f"{name} must be non-empty.")
    for i, x in enumerate(data):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(f"All {name} elements must be numeric. Got {type(x).__name__} at index {i}.")
        if isinstance(x, float) and math.isnan(x):
            raise ValueError(
                f"{name} contains NaN at index {i}. Sorting-based statistics (median, "
                "percentile, mode) are undefined for NaN since comparisons with NaN are "
                "always False; remove or impute NaN values before analysis."
            )


class DescriptiveStats:
    """Computes descriptive statistics for a single numeric dataset.

    Parameters
    ----------
    data : Sequence[int or float]
        Non-empty sequence of numeric values (bools and NaN rejected).

    Raises
    ------
    ValueError
        If ``data`` is empty or contains NaN.
    TypeError
        If any element is non-numeric.
    """

    def __init__(self, data: Sequence[Number]):
        _validate_numeric_data(data)
        self._raw: List[Number] = list(data)
        self.data: List[Number] = sorted(self._raw)
        self.n: int = len(self.data)

    def __repr__(self) -> str:
        preview = self.data[:5]
        tail = "" if self.n <= 5 else "..."
        return f"DescriptiveStats(n={self.n}, data={preview}{tail})"

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return self.n


    # Central tendency

    def mean(self) -> float:
        """Arithmetic mean."""
        return sum(self.data) / self.n

    def median(self) -> float:
        """50th percentile (linear interpolation, matches ``numpy.median``)."""
        return self.percentile(50)

    def mode(self) -> List[Number]:
        """All values tied for highest frequency, ascending.

        Note
        ----
        For continuous data with no repeated values, every value has
        frequency 1, so this returns the *entire* sorted dataset -- that
        is mathematically correct but rarely useful; mode is best suited
        to discrete/categorical-like data.
        """
        counts = Counter(self.data)
        max_freq = max(counts.values())
        return sorted(k for k, v in counts.items() if v == max_freq)

    def percentile(self, p: float) -> float:
        """The ``p``-th percentile via linear interpolation (numpy's default 'linear' method).

        Raises
        ------
        ValueError
            If ``p`` is not in ``[0, 100]``.
        """
        if not (0 <= p <= 100):
            raise ValueError("Percentile must be between 0 and 100.")
        k = (p / 100.0) * (self.n - 1)
        i = int(k)
        f = k - i
        if i >= self.n - 1:
            return float(self.data[-1])
        return self.data[i] + f * (self.data[i + 1] - self.data[i])


    # Spread

    def variance(self, ddof: int = 0) -> float:
        """Variance. ``ddof=0`` for population, ``ddof=1`` for sample (Bessel's correction).

        Raises
        ------
        ValueError
            If ``ddof`` is not 0 or 1, or ``ddof=1`` with fewer than 2 points.
        """
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1.")
        if ddof == 1 and self.n < 2:
            raise ValueError("Sample variance requires at least 2 data points.")
        mu = self.mean()
        sse = sum((x - mu) ** 2 for x in self.data)
        return sse / (self.n - ddof)

    def std(self, ddof: int = 0) -> float:
        """Standard deviation; see :meth:`variance` for ``ddof`` semantics."""
        return math.sqrt(self.variance(ddof))

    def data_range(self) -> float:
        """``max - min``."""
        return float(self.data[-1] - self.data[0])

    def iqr(self) -> float:
        """Interquartile range: ``P75 - P25``."""
        return self.percentile(75) - self.percentile(25)


    # Shape

    def skewness(self) -> float:
        """Population (biased) Fisher-Pearson skewness coefficient.

        Returns 0.0 for a zero-variance (constant) dataset rather than
        raising, since skewness of a degenerate distribution is
        conventionally taken to be 0.
        """
        mu = self.mean()
        sigma = self.std(ddof=0)
        if sigma == 0.0:
            return 0.0
        return (sum((x - mu) ** 3 for x in self.data) / self.n) / (sigma ** 3)

    def kurtosis(self) -> float:
        """Population (biased) excess kurtosis (normal distribution -> 0.0).

        Returns 0.0 for a zero-variance (constant) dataset rather than
        raising.
        """
        mu = self.mean()
        sigma = self.std(ddof=0)
        if sigma == 0.0:
            return 0.0
        return (sum((x - mu) ** 4 for x in self.data) / self.n) / (sigma ** 4) - 3.0


    # Standardization

    def z_scores(self, ddof: int = 0) -> List[float]:
        """Standardized values ``(x - mean) / std``, in the *original* (unsorted) input order.

        Raises
        ------
        ValueError
            If the standard deviation is zero (constant data).
        """
        mu = self.mean()
        sigma = self.std(ddof=ddof)
        if sigma == 0.0:
            raise ValueError("Cannot compute z-scores: standard deviation is zero (constant data).")
        return [(x - mu) / sigma for x in self._raw]


    # Bivariate / multivariate (static methods)

    @staticmethod
    def covariance(x: Sequence[Number], y: Sequence[Number], ddof: int = 0) -> float:
        """Sample or population covariance between two equal-length sequences.

        Raises
        ------
        ValueError
            If lengths differ, either is empty, ``ddof`` is invalid, or
            ``ddof=1`` with fewer than 2 points.
        """
        _validate_numeric_data(x, "x")
        _validate_numeric_data(y, "y")
        if len(x) != len(y):
            raise ValueError("Lists must have the same length.")
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1.")
        n = len(x)
        if ddof == 1 and n < 2:
            raise ValueError("Sample covariance requires at least 2 data points.")
        mx = sum(x) / n
        my = sum(y) / n
        cross = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        return cross / (n - ddof)

    @staticmethod
    def correlation(x: Sequence[Number], y: Sequence[Number]) -> float:
        """Pearson correlation coefficient. Returns 0.0 if either series is constant.

        Raises
        ------
        ValueError
            If lengths differ or fewer than 2 points are given.
        """
        _validate_numeric_data(x, "x")
        _validate_numeric_data(y, "y")
        if len(x) != len(y):
            raise ValueError("Lists must have the same length.")
        if len(x) < 2:
            raise ValueError("Need at least 2 points.")
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)
        sx = (sum((xi - mx) ** 2 for xi in x) / (n - 1)) ** 0.5
        sy = (sum((yi - my) ** 2 for yi in y) / (n - 1)) ** 0.5
        if sx == 0.0 or sy == 0.0:
            return 0.0
        return cov / (sx * sy)

    @staticmethod
    def covariance_matrix(dataset: Sequence[Sequence[Number]], ddof: int = 1) -> "Matrix":
        """Symmetric p x p covariance matrix for ``p`` variables.

        Parameters
        ----------
        dataset : Sequence[Sequence[Number]]
            ``dataset[i]`` is the ``i``-th variable's observations; all
            variables must have the same length.
        ddof : int, optional
            Delta degrees of freedom (default 1, sample covariance).

        Notes
        -----
        Exploits symmetry (computes each off-diagonal pair once) rather
        than recomputing ``covariance(dataset[j], dataset[i])`` separately
        from ``covariance(dataset[i], dataset[j])`` -- halving the work
        compared to a naive double loop.
        """
        p = len(dataset)
        if p == 0:
            raise ValueError("Dataset must have at least one variable.")
        for v in dataset:
            _validate_numeric_data(v, "each variable in dataset")

        mat = Matrix.zeros(p, p)
        for i in range(p):
            for j in range(i, p):
                cov_ij = DescriptiveStats.covariance(dataset[i], dataset[j], ddof=ddof)
                mat.rows[i].components[j] = cov_ij
                mat.rows[j].components[i] = cov_ij
        return mat

    @staticmethod
    def correlation_matrix(dataset: Sequence[Sequence[Number]]) -> "Matrix":
        """Symmetric p x p Pearson correlation matrix for ``p`` variables.

        See :meth:`covariance_matrix` for the symmetry-exploiting
        optimization; the diagonal is set to exactly ``1.0`` rather than
        computed (avoids a division-by-zero edge case for constant
        variables, and is exact by definition).
        """
        p = len(dataset)
        if p == 0:
            raise ValueError("Dataset must have at least one variable.")
        for v in dataset:
            _validate_numeric_data(v, "each variable in dataset")

        mat = Matrix.zeros(p, p)
        for i in range(p):
            mat.rows[i].components[i] = 1.0
            for j in range(i + 1, p):
                corr_ij = DescriptiveStats.correlation(dataset[i], dataset[j])
                mat.rows[i].components[j] = corr_ij
                mat.rows[j].components[i] = corr_ij
        return mat


    # Reporting

    def summary(self) -> str:
        """A formatted multi-line summary of the key descriptive statistics."""
        lines = [
            "Descriptive Statistics Summary",
            "\u2500" * 35,
            f"  {'Count:':<12} {self.n}",
            f"  {'Mean:':<12} {self.mean():.4f}",
            f"  {'Median:':<12} {self.median():.4f}",
            f"  {'Std Dev:':<12} {self.std(ddof=0):.4f}",
            f"  {'Variance:':<12} {self.variance(ddof=0):.4f}",
            f"  {'Min:':<12} {self.data[0]}",
            f"  {'Max:':<12} {self.data[-1]}",
            f"  {'Range:':<12} {self.data_range():.4f}",
            f"  {'IQR:':<12} {self.iqr():.4f}",
            f"  {'Skewness:':<12} {self.skewness():.4f}",
            f"  {'Kurtosis:':<12} {self.kurtosis():.4f}",
            "\u2500" * 35,
        ]
        return "\n".join(lines)
