import math
from collections import Counter


class DescriptiveStats:
    def __init__(self, data: list):
        if not data:
            raise ValueError("Data must be non‑empty.")
        self.data = sorted(data)           
        self.n = len(data)

    def __repr__(self):
        return f"DescriptiveStats(n={self.n})"

    def __iter__(self):
        return iter(self.data)
 
    def mean(self) -> float:
        return sum(self.data) / self.n

    def median(self) -> float:
        return self.percentile(50)         

    def mode(self) -> list:
        counts = Counter(self.data)
        if not counts:
            return []
        max_freq = max(counts.values())
        return [k for k, freq in counts.items() if freq == max_freq]

    def percentile(self, p: float) -> float:
        if not (0 <= p <= 100):
            raise ValueError("Percentile must be between 0 and 100.")

        k = (p / 100.0) * (self.n - 1)
        i = int(k)          
        f = k - i          
        if i >= self.n - 1:
            return self.data[-1]
        return self.data[i] + f * (self.data[i+1] - self.data[i])

    def variance(self, ddof: int = 0) -> float:
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1.")
        mu = self.mean()
        sse = sum((x - mu) ** 2 for x in self.data)   
        if ddof == 0:
            return sse / self.n
        else:  
            if self.n < 2:
                raise ValueError("Sample variance requires at least 2 data points.")
            return sse / (self.n - 1)                  

    def std(self, ddof: int = 0) -> float:
        return math.sqrt(self.variance(ddof))

    def data_range(self) -> float:                  
        return self.data[-1] - self.data[0]           

    def iqr(self) -> float:
        return self.percentile(75) - self.percentile(25)

    def skewness(self) -> float:
        mu = self.mean()
        sigma = self.std(ddof=0)        
        if sigma == 0:
            return 0.0                  
        n = self.n
        sum_cubed = sum((x - mu) ** 3 for x in self.data)
        return (sum_cubed / n) / (sigma ** 3)

    def kurtosis(self) -> float:
        mu = self.mean()
        sigma = self.std(ddof=0)
        if sigma == 0:
            return 0.0
        n = self.n
        sum_fourth = sum((x - mu) ** 4 for x in self.data)
        return (sum_fourth / n) / (sigma ** 4) - 3.0



