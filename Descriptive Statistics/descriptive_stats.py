import math

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