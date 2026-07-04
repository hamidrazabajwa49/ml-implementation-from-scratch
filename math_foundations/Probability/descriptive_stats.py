import math
from collections import Counter
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Matrix.matrix import Matrix


class DescriptiveStats:
    def __init__(self, data: list):
        if not data:
            raise ValueError("Data must be non-empty.")
        for i, x in enumerate(data):
            if not isinstance(x, (int, float)):
                raise TypeError(f"All data elements must be numeric. Got {type(x).__name__} at index {i}.")
        self._raw = data
        self.data = sorted(data)
        self.n = len(self.data)

    def __repr__(self) -> str:
        preview = self.data[:5]
        tail = "" if self.n <= 5 else "..."
        return f"DescriptiveStats(n={self.n}, data={preview}{tail})"

    def __iter__(self):
        return iter(self.data)

    def mean(self) -> float:
        return sum(self.data) / self.n

    def median(self) -> float:
        return self.percentile(50)

    def mode(self) -> list:
        counts = Counter(self.data)
        max_freq = max(counts.values())
        return sorted([k for k, v in counts.items() if v == max_freq])

    def percentile(self, p: float) -> float:
        if not (0 <= p <= 100):
            raise ValueError("Percentile must be between 0 and 100.")
        k = (p / 100.0) * (self.n - 1)
        i = int(k)
        f = k - i
        if i >= self.n - 1:
            return float(self.data[-1])
        return self.data[i] + f * (self.data[i + 1] - self.data[i])

    def variance(self, ddof: int = 0) -> float:
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1.")
        if ddof == 1 and self.n < 2:
            raise ValueError("Sample variance requires at least 2 data points.")
        mu = self.mean()
        sse = sum((x - mu) ** 2 for x in self.data)
        return sse / (self.n - ddof)

    def std(self, ddof: int = 0) -> float:
        return math.sqrt(self.variance(ddof))

    def data_range(self) -> float:
        return float(self.data[-1] - self.data[0])

    def iqr(self) -> float:
        return self.percentile(75) - self.percentile(25)

    def skewness(self) -> float:
        mu = self.mean()
        sigma = self.std(ddof=0)
        if sigma == 0.0:
            return 0.0
        return (sum((x - mu) ** 3 for x in self.data) / self.n) / (sigma ** 3)

    def kurtosis(self) -> float:
        mu = self.mean()
        sigma = self.std(ddof=0)
        if sigma == 0.0:
            return 0.0
        return (sum((x - mu) ** 4 for x in self.data) / self.n) / (sigma ** 4) - 3.0

    @staticmethod
    def covariance(x: list, y: list, ddof: int = 0) -> float:
        if len(x) != len(y):
            raise ValueError("Lists must have the same length.")
        if len(x) == 0:
            raise ValueError("Lists must be non-empty.")
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
    def correlation(x: list, y: list) -> float:
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
    def covariance_matrix(dataset: list) -> 'Matrix':
        p = len(dataset)
        if p == 0:
            raise ValueError("Dataset must have at least one variable.")
        mat = Matrix.zeros(p, p)
        for i in range(p):
            for j in range(p):
                mat.rows[i].components[j] = DescriptiveStats.covariance(dataset[i], dataset[j], ddof=1)
        return mat

    @staticmethod
    def correlation_matrix(dataset: list) -> 'Matrix':
        p = len(dataset)
        if p == 0:
            raise ValueError("Dataset must have at least one variable.")
        mat = Matrix.zeros(p, p)
        for i in range(p):
            for j in range(p):
                mat.rows[i].components[j] = DescriptiveStats.correlation(dataset[i], dataset[j])
        return mat

    def summary(self) -> str:
        lines = [
            "Descriptive Statistics Summary",
            "─" * 35,
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
            "─" * 35,
        ]
        return "\n".join(lines)
