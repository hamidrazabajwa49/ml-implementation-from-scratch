import math
import os
import sys


class Distribution:

    def mean(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError

    def std(self) -> float:
        return math.sqrt(self.variance())

    def sample(self, num_samples: int = 1) -> list:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def summary(self) -> str:
        return (
            f"{self.__class__.__name__}: "
            f"mean={self.mean():.4f}, "
            f"variance={self.variance():.4f}, "
            f"std={self.std():.4f}"
        )
