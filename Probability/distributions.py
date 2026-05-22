import os
import sys

# Maintain your path fix so Probability can talk to Matrix/Vectors if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

import math

class Distribution:
  """Abstract base class for probability distributions."""

  def mean(self) -> float:
      """Return the expected value (mean)."""
      raise NotImplementedError

  def variance(self) -> float:
      """Return the variance."""
      raise NotImplementedError

  def std(self) -> float:
      """Return the standard deviation (√variance)."""
      return math.sqrt(self.variance())

  def sample(self, num_samples: int = 1) -> list:
      """Generate random samples from the distribution."""
      raise NotImplementedError

  def __repr__(self) -> str:
      params = ", ".join(f"{k}={v}" for k, v in self._params.items())
      return f"{self.__class__.__name__}({params})"

  def summary(self) -> str:
      """Return a one‑line summary with mean, variance, and std."""
      return (
          f"{self.__class__.__name__}: "
          f"mean={self.mean():.4f}, "
          f"variance={self.variance():.4f}, "
          f"std={self.std():.4f}"
      )
