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
